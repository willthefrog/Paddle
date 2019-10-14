# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import print_function

import six
from ..framework import Parameter
from .. import core
from .. import unique_name
# potential circular import when python version is old
try:
    from .. import layers
except ImportError:
    import sys
    layers = sys.modules['paddle.fluid.layers']

__all__ = ['mixed_precision_global_state', 'mixed_precision_context',
           'StaticLossScale', 'DynamicLossScale',
           'update_loss_scale', 'scale_gradient']

_mixed_precision_global_state = None


def mixed_precision_global_state():
    return _mixed_precision_global_state


class LossScale(object):
    def __init__(self):
        super(LossScale, self).__init__()

    def get_loss_scale_var(self):
        return self.scale

    def increment(self):
        raise NotImplementedError()

    def decrement(self):
        raise NotImplementedError()


class StaticLossScale(LossScale):
    """
    Static (fixed) loss scale manager.

    Args:
        init_loss_scale (float): initial loss scale value.

    Examples:

        .. code-block:: python

            from paddle import fluid
            from paddle.fluid.mixed_precision import (mixed_precision_context,
                                                      StaticLossScale)

            with mixed_precision_context(StaticLossScale(8.), True) as ctx:
                # ...
                # scale loss
                loss_scale = ctx.get_loss_scale_var()

    """

    def __init__(self, init_loss_scale=1.):
        super(StaticLossScale, self).__init__()
        self.scale = layers.create_global_var(
            name=unique_name.generate("loss_scale"),
            shape=[1],
            value=init_loss_scale,
            dtype='float32',
            persistable=True)


class DynamicLossScale(LossScale):
    """
    Dynamic loss scale manager. it works as follows:
    if gradients is valid for `increment_every` steps, loss scale values is
    increased by `factor`, otherwise loss scale values is decreased by `factor`

    Args:
        init_loss_scale (float): initial loss scale value.
        increment_every (int): minimum 'good' steps before increase loss scale.
        factor (float): increase/decrease loss scale by this much.

    Examples:

        .. code-block:: python

            from paddle import fluid
            from paddle.fluid.mixed_precision import (mixed_precision_context,
                                                      StaticLossScale)

            loss_scale = DynamicLossScale(8., 1000, 4.)
            with mixed_precision_context(loss_scale, True) as ctx:
                # ...
                # scale loss
                loss_scale = ctx.get_loss_scale_var()

    """

    def __init__(self, init_loss_scale=2**15, increment_every=2000, factor=2.):
        super(DynamicLossScale, self).__init__()
        self.scale = layers.create_global_var(
            name=unique_name.generate("loss_scale"),
            shape=[1],
            value=init_loss_scale,
            dtype='float32',
            persistable=True)
        self.good_steps = layers.create_global_var(
            name=unique_name.generate("good_steps"),
            shape=[1],
            value=0,
            dtype='int32',
            persistable=True)
        self.increment_every = layers.fill_constant(
            shape=[1], dtype='int32', value=increment_every)
        self.factor = factor

    def increment(self):
        enough_steps = layers.less_than(self.increment_every,
                                        self.good_steps + 1)
        with layers.Switch() as switch:
            with switch.case(enough_steps):
                new_scale = self.scale * self.factor
                scale_valid = layers.isfinite(new_scale)
                with layers.Switch() as switch2:
                    with switch2.case(scale_valid):
                        layers.assign(new_scale, self.scale)
                        layers.assign(layers.zeros_like(self.good_steps),
                                      self.good_steps)
                    with switch2.default():
                        layers.increment(self.good_steps)
            with switch.default():
                layers.increment(self.good_steps)

    def decrement(self):
        new_scale = self.scale / self.factor
        one = layers.fill_constant(shape=[1], dtype='float32', value=1.0)
        less_than_one = layers.less_than(new_scale, one)
        with layers.Switch() as switch:
            with switch.case(less_than_one):
                layers.assign(one, self.scale)
            with switch.default():
                layers.assign(new_scale, self.scale)

        layers.assign(layers.zeros_like(self.good_steps),
                      self.good_steps)


class mixed_precision_context(object):
    """
    Context manager for mixed precision training.

    Args:
        loss_scale (float, str or obj): loss scale settings, can be:
            1. an number: use fixed loss scale.
            2. 'dynamic': use a default `DynamicLossScale`.
            3. `DynamicLossScale` or `StaticLossScale` instance.
         enabled (bool): enable mixed precision training.

    Examples:

        .. code-block:: python

            from paddle import fluid
            from paddle.fluid.mixed_precision import mixed_precision_context

            with mixed_precision_context('dynamic', True) as ctx:
                # cast inputs to float16
                inputs = fluid.layers.cast(inputs, "float16")
                # build model here
                logits = model(inputs)
                # use float32 for softmax
                logits = fluid.layers.cast(logits, "float32")
                softmax = fluid.layers.softmax(logits)
                loss = fluid.layers.cross_entropy(input=softmax, label=label)
                avg_loss = fluid.layers.mean(loss)
                # scale loss
                loss_scale = ctx.get_loss_scale_var()
                avg_loss *= loss_scale
                optimizer = fluid.optimizer.Momentum(...)
                optimizer.minimize(avg_loss)

    """

    def __init__(self, loss_scale=1., enabled=True):
        super(mixed_precision_context, self).__init__()
        self.enabled = enabled
        if not enabled:
            return
        if isinstance(loss_scale, six.integer_types + (float,)):
            self.loss_scale = StaticLossScale(loss_scale)
        elif loss_scale == 'dynamic':
            self.loss_scale = DynamicLossScale()
        else:
            assert isinstance(loss_scale, LossScale), \
                "Invalid loss scale argument"
            self.loss_scale = loss_scale

    @property
    def dynamic_scaling(self):
        return isinstance(self.loss_scale, DynamicLossScale)

    def __getattr__(self, attr):
        if attr in ['get_loss_scale_var', 'increment', 'decrement']:
            return getattr(self.loss_scale, attr)

    def __enter__(self):
        if not self.enabled:
            return
        global _mixed_precision_global_state
        _mixed_precision_global_state = self
        return mixed_precision_global_state()

    def __exit__(self, *args):
        if not self.enabled:
            return
        global _mixed_precision_global_state
        _mixed_precision_global_state = None
        return mixed_precision_global_state()


def scale_gradient(block, context):
    state = mixed_precision_global_state()
    if state is None:
        return
    scale = state.get_loss_scale_var()
    op_desc = block.desc.op(block.desc.op_size() - 1)
    op_role_attr_name = core.op_proto_and_checker_maker.kOpRoleAttrName()
    bwd_role = core.op_proto_and_checker_maker.OpRole.Backward
    for name in [n for n in op_desc.output_arg_names() if n in context]:
        fwd_var = block._var_recursive(context[name])
        if not isinstance(fwd_var, Parameter):
            continue  # TODO verify all use cases
        clip_op_desc = block.desc.append_op()
        clip_op_desc.set_type("elementwise_div")
        clip_op_desc.set_input("X", [name])
        clip_op_desc.set_input("Y", [scale.name])
        clip_op_desc.set_output("Out", [name])
        clip_op_desc._set_attr(op_role_attr_name, bwd_role)


def update_loss_scale(grads):
    state = mixed_precision_global_state()
    if state is None or not state.dynamic_scaling:
        return

    per_grad_check = layers.stack([layers.reduce_sum(g) for g in grads])
    grad_valid = layers.isfinite(per_grad_check)

    with layers.Switch() as switch:
        with switch.case(grad_valid):
            state.increment()
        with switch.default():
            state.decrement()
    return grad_valid
