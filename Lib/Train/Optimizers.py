# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Rectified Adam (RAdam) optimizer."""
import tensorflow as tf
import warnings
import abc
from typing import Union, Callable, Dict
from typeguard import typechecked
import numpy as np

Number = Union[
    float,
    int,
    np.float16,
    np.float32,
    np.float64,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
]

Initializer = Union[None, dict, str, Callable,
                    tf.keras.initializers.Initializer]
Regularizer = Union[None, dict, str, Callable,
                    tf.keras.regularizers.Regularizer]
Constraint = Union[None, dict, str, Callable, tf.keras.constraints.Constraint]
Activation = Union[None, str, Callable]
Optimizer = Union[tf.keras.optimizers.Optimizer, str]

# TensorLike = Union[
#     List[Union[Number, list]],
#     tuple,
#     Number,
#     np.ndarray,
#     tf.Tensor,
#     tf.SparseTensor,
#     tf.Variable,
#     keras_tensor.KerasTensor,
# ]
FloatTensorLike = Union[tf.Tensor, float, np.float16, np.float32, np.float64]
AcceptableDTypes = Union[tf.DType, np.dtype, type, int, str, None]


class RectifiedAdam(tf.keras.optimizers.Optimizer):

    @typechecked
    def __init__(
        self,
        learning_rate: Union[FloatTensorLike, Callable, Dict] = 0.001,
        beta_1: FloatTensorLike = 0.9,
        beta_2: FloatTensorLike = 0.999,
        epsilon: FloatTensorLike = 1e-7,
        weight_decay: Union[FloatTensorLike, Callable, Dict] = 0.0,
        amsgrad: bool = False,
        sma_threshold: FloatTensorLike = 5.0,
        total_steps: int = 0,
        warmup_proportion: FloatTensorLike = 0.1,
        min_lr: FloatTensorLike = 0.0,
        name: str = "RectifiedAdam",
        **kwargs,
    ):
        super().__init__(name, **kwargs)

        if isinstance(learning_rate, Dict):
            learning_rate = tf.keras.optimizers.schedules.deserialize(
                learning_rate)

        if isinstance(weight_decay, Dict):
            weight_decay = tf.keras.optimizers.schedules.deserialize(
                weight_decay)

        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("beta_1", beta_1)
        self._set_hyper("beta_2", beta_2)
        self._set_hyper("decay", self._initial_decay)
        self._set_hyper("weight_decay", weight_decay)
        self._set_hyper("sma_threshold", sma_threshold)
        self._set_hyper("total_steps", float(total_steps))
        self._set_hyper("warmup_proportion", warmup_proportion)
        self._set_hyper("min_lr", min_lr)
        self.epsilon = epsilon or tf.keras.backend.epsilon()
        self.amsgrad = amsgrad
        self._has_weight_decay = weight_decay != 0.0
        self._initial_total_steps = total_steps

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "m")
        for var in var_list:
            self.add_slot(var, "v")
        if self.amsgrad:
            for var in var_list:
                self.add_slot(var, "vhat")

    def set_weights(self, weights):
        params = self.weights
        num_vars = int((len(params) - 1) / 2)
        if len(weights) == 3 * num_vars + 1:
            weights = weights[: len(params)]
        super().set_weights(weights)

    def _decayed_wd(self, var_dtype):
        wd_t = self._get_hyper("weight_decay", var_dtype)
        if isinstance(wd_t, tf.keras.optimizers.schedules.LearningRateSchedule):
            wd_t = tf.cast(wd_t(self.iterations), var_dtype)
        return wd_t

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super()._prepare_local(var_device, var_dtype, apply_state)
        lr_t = self._decayed_lr(var_dtype)
        wd_t = self._decayed_wd(var_dtype)
        beta_1_t = self._get_hyper("beta_1", var_dtype)
        beta_2_t = self._get_hyper("beta_2", var_dtype)
        local_step = tf.cast(self.iterations + 1, var_dtype)
        beta_1_power = tf.pow(beta_1_t, local_step)
        beta_2_power = tf.pow(beta_2_t, local_step)
        one_minus_beta_1_t = 1.0 - beta_1_t
        recip_one_minus_beta_1_power = 1.0 / (1.0 - beta_1_power)
        one_minus_beta_2_t = 1.0 - beta_2_t
        recip_one_minus_beta_2_power = 1.0 / (1.0 - beta_2_power)
        sma_inf = 2.0 / one_minus_beta_2_t - 1.0
        sma_t = sma_inf - 2.0 * local_step * beta_2_power * recip_one_minus_beta_2_power
        r_t = tf.sqrt(
            (sma_t - 4.0)
            / (sma_inf - 4.0)
            * (sma_t - 2.0)
            / (sma_inf - 2.0)
            * sma_inf
            / sma_t
        )
        sma_threshold = self._get_hyper("sma_threshold", var_dtype)
        sma_t_ge_sma_threshold = sma_t >= sma_threshold
        if self._initial_total_steps > 0:
            total_steps = self._get_hyper("total_steps", var_dtype)
            warmup_steps = total_steps * \
                self._get_hyper("warmup_proportion", var_dtype)
            min_lr = self._get_hyper("min_lr", var_dtype)
            decay_steps = tf.maximum(total_steps - warmup_steps, 1)
            decay_rate = (min_lr - lr_t) / decay_steps
            lr_t = tf.where(
                local_step <= warmup_steps,
                lr_t * (local_step / warmup_steps),
                lr_t + decay_rate *
                tf.minimum(local_step - warmup_steps, decay_steps),
            )
        apply_state[(var_device, var_dtype)].update(
            dict(
                lr_t=lr_t,
                wd_t=wd_t,
                beta_1_t=beta_1_t,
                beta_2_t=beta_2_t,
                epsilon_t=tf.convert_to_tensor(self.epsilon, var_dtype),
                local_step=local_step,
                beta_1_power=beta_1_power,
                beta_2_power=beta_2_power,
                sma_inf=sma_inf,
                sma_t=sma_t,
                one_minus_beta_1_t=one_minus_beta_1_t,
                recip_one_minus_beta_1_power=recip_one_minus_beta_1_power,
                one_minus_beta_2_t=one_minus_beta_2_t,
                recip_one_minus_beta_2_power=recip_one_minus_beta_2_power,
                r_t=r_t,
                sma_t_ge_sma_threshold=sma_t_ge_sma_threshold,
            )
        )

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coef = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")

        m_t = m.assign(
            coef["beta_1_t"] * m + coef["one_minus_beta_1_t"] * grad,
            use_locking=self._use_locking,
        )
        m_corr_t = m_t * coef["recip_one_minus_beta_1_power"]

        v_t = v.assign(
            coef["beta_2_t"] * v +
            coef["one_minus_beta_2_t"] * tf.square(grad),
            use_locking=self._use_locking,
        )
        if self.amsgrad:
            vhat = self.get_slot(var, "vhat")
            vhat_t = vhat.assign(tf.maximum(vhat, v_t),
                                 use_locking=self._use_locking)
            v_corr_t = tf.sqrt(vhat_t * coef["recip_one_minus_beta_2_power"])
        else:
            vhat_t = None
            v_corr_t = tf.sqrt(v_t * coef["recip_one_minus_beta_2_power"])

        var_t = tf.where(
            coef["sma_t_ge_sma_threshold"],
            coef["r_t"] * m_corr_t / (v_corr_t + coef["epsilon_t"]),
            m_corr_t,
        )

        if self._has_weight_decay:
            var_t += coef["wd_t"] * var

        var_update = var.assign_sub(
            coef["lr_t"] * var_t, use_locking=self._use_locking)

        updates = [var_update, m_t, v_t]
        if self.amsgrad:
            updates.append(vhat_t)
        return tf.group(*updates)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coef = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)

        m = self.get_slot(var, "m")
        m_scaled_g_values = grad * coef["one_minus_beta_1_t"]
        m_t = m.assign(m * coef["beta_1_t"], use_locking=self._use_locking)
        with tf.control_dependencies([m_t]):
            m_t = self._resource_scatter_add(m, indices, m_scaled_g_values)
        m_corr_t = m_t * coef["recip_one_minus_beta_1_power"]

        v = self.get_slot(var, "v")
        v_scaled_g_values = (grad * grad) * coef["one_minus_beta_2_t"]
        v_t = v.assign(v * coef["beta_2_t"], use_locking=self._use_locking)
        with tf.control_dependencies([v_t]):
            v_t = self._resource_scatter_add(v, indices, v_scaled_g_values)

        if self.amsgrad:
            vhat = self.get_slot(var, "vhat")
            vhat_t = vhat.assign(tf.maximum(vhat, v_t),
                                 use_locking=self._use_locking)
            v_corr_t = tf.sqrt(vhat_t * coef["recip_one_minus_beta_2_power"])
        else:
            vhat_t = None
            v_corr_t = tf.sqrt(v_t * coef["recip_one_minus_beta_2_power"])

        var_t = tf.where(
            coef["sma_t_ge_sma_threshold"],
            coef["r_t"] * m_corr_t / (v_corr_t + coef["epsilon_t"]),
            m_corr_t,
        )

        if self._has_weight_decay:
            var_t += coef["wd_t"] * var

        with tf.control_dependencies([var_t]):
            var_update = self._resource_scatter_add(
                var, indices, tf.gather(-coef["lr_t"] * var_t, indices)
            )

        updates = [var_update, m_t, v_t]
        if self.amsgrad:
            updates.append(vhat_t)
        return tf.group(*updates)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter("learning_rate"),
                "beta_1": self._serialize_hyperparameter("beta_1"),
                "beta_2": self._serialize_hyperparameter("beta_2"),
                "decay": self._serialize_hyperparameter("decay"),
                "weight_decay": self._serialize_hyperparameter("weight_decay"),
                "sma_threshold": self._serialize_hyperparameter("sma_threshold"),
                "epsilon": self.epsilon,
                "amsgrad": self.amsgrad,
                "total_steps": int(self._serialize_hyperparameter("total_steps")),
                "warmup_proportion": self._serialize_hyperparameter(
                    "warmup_proportion"
                ),
                "min_lr": self._serialize_hyperparameter("min_lr"),
            }
        )
        return config


class Lookahead(tf.keras.optimizers.Optimizer):
    """This class allows to extend optimizers with the lookahead mechanism.
    The mechanism is proposed by Michael R. Zhang et.al in the paper
    [Lookahead Optimizer: k steps forward, 1 step back]
    (https://arxiv.org/abs/1907.08610v1). The optimizer iteratively updates two
    sets of weights: the search directions for weights are chosen by the inner
    optimizer, while the "slow weights" are updated each `k` steps based on the
    directions of the "fast weights" and the two sets of weights are
    synchronized. This method improves the learning stability and lowers the
    variance of its inner optimizer.
    Example of usage:
    ```python
    opt = tf.keras.optimizers.SGD(learning_rate)
    opt = tfa.optimizers.Lookahead(opt)
    ```
    """

    @typechecked
    def __init__(
        self,
        optimizer: Optimizer,
        sync_period: int = 6,
        slow_step_size: FloatTensorLike = 0.5,
        name: str = "Lookahead",
        **kwargs,
    ):
        r"""Wrap optimizer with the lookahead mechanism.
        Args:
            optimizer: The original optimizer that will be used to compute
                and apply the gradients.
            sync_period: An integer. The synchronization period of lookahead.
                Enable lookahead mechanism by setting it with a positive value.
            slow_step_size: A floating point value.
                The ratio for updating the slow weights.
            name: Optional name for the operations created when applying
                gradients. Defaults to "Lookahead".
            **kwargs: keyword arguments. Allowed to be {`clipnorm`,
                `clipvalue`, `lr`, `decay`}. `clipnorm` is clip gradients
                by norm; `clipvalue` is clip gradients by value, `decay` is
                included for backward compatibility to allow time inverse
                decay of learning rate. `lr` is included for backward
                compatibility, recommended to use `learning_rate` instead.
        """
        super().__init__(name, **kwargs)

        if isinstance(optimizer, str):
            optimizer = tf.keras.optimizers.get(optimizer)
        if not isinstance(optimizer, tf.keras.optimizers.Optimizer):
            raise TypeError(
                "optimizer is not an object of tf.keras.optimizers.Optimizer"
            )

        self._optimizer = optimizer
        self._set_hyper("sync_period", sync_period)
        self._set_hyper("slow_step_size", slow_step_size)
        self._initialized = False
        self._track_trackable(self._optimizer, "lh_base_optimizer")

    def _create_slots(self, var_list):
        self._optimizer._create_slots(
            var_list=var_list
        )  # pylint: disable=protected-access
        for var in var_list:
            self.add_slot(var, "slow", initializer=var)

    def _create_hypers(self):
        self._optimizer._create_hypers()  # pylint: disable=protected-access

    def _prepare(self, var_list):
        return self._optimizer._prepare(
            var_list=var_list
        )  # pylint: disable=protected-access

    def apply_gradients(self, grads_and_vars, name=None, **kwargs):
        self._optimizer._iterations = (
            self.iterations
        )  # pylint: disable=protected-access
        return super().apply_gradients(grads_and_vars, name, **kwargs)

    def _look_ahead_op(self, var):
        var_dtype = var.dtype.base_dtype
        slow_var = self.get_slot(var, "slow")
        local_step = tf.cast(self.iterations + 1, tf.dtypes.int64)
        sync_period = self._get_hyper("sync_period", tf.dtypes.int64)
        slow_step_size = self._get_hyper("slow_step_size", var_dtype)
        step_back = slow_var + slow_step_size * (var - slow_var)
        sync_cond = tf.equal(
            tf.math.floordiv(local_step, sync_period) * sync_period, local_step
        )
        with tf.control_dependencies([step_back]):
            slow_update = slow_var.assign(
                tf.where(sync_cond, step_back, slow_var), use_locking=self._use_locking
            )
            var_update = var.assign(
                tf.where(sync_cond, step_back, var), use_locking=self._use_locking
            )
        return tf.group(slow_update, var_update)

    @property
    def weights(self):
        return self._weights + self._optimizer.weights

    def _resource_apply_dense(self, grad, var):
        train_op = self._optimizer._resource_apply_dense(
            grad, var
        )  # pylint: disable=protected-access
        with tf.control_dependencies([train_op]):
            look_ahead_op = self._look_ahead_op(var)
        return tf.group(train_op, look_ahead_op)

    def _resource_apply_sparse(self, grad, var, indices):
        train_op = (
            self._optimizer._resource_apply_sparse(  # pylint: disable=protected-access
                grad, var, indices
            )
        )
        with tf.control_dependencies([train_op]):
            look_ahead_op = self._look_ahead_op(var)
        return tf.group(train_op, look_ahead_op)

    def get_config(self):
        config = {
            "optimizer": tf.keras.optimizers.serialize(self._optimizer),
            "sync_period": self._serialize_hyperparameter("sync_period"),
            "slow_step_size": self._serialize_hyperparameter("slow_step_size"),
        }
        base_config = super().get_config()
        return {**base_config, **config}

    @property
    def learning_rate(self):
        return self._optimizer._get_hyper("learning_rate")

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        self._optimizer._set_hyper("learning_rate", learning_rate)

    @property
    def lr(self):
        return self.learning_rate

    @lr.setter
    def lr(self, lr):
        self.learning_rate = lr

    @classmethod
    def from_config(cls, config, custom_objects=None):
        optimizer = tf.keras.optimizers.deserialize(
            config.pop("optimizer"), custom_objects=custom_objects
        )
        return cls(optimizer, **config)


class DecoupledWeightDecayExtension:
    """This class allows to extend optimizers with decoupled weight decay.
    It implements the decoupled weight decay described by Loshchilov & Hutter
    (https://arxiv.org/pdf/1711.05101.pdf), in which the weight decay is
    decoupled from the optimization steps w.r.t. to the loss function.
    For SGD variants, this simplifies hyperparameter search since it decouples
    the settings of weight decay and learning rate.
    For adaptive gradient algorithms, it regularizes variables with large
    gradients more than L2 regularization would, which was shown to yield
    better training loss and generalization error in the paper above.
    This class alone is not an optimizer but rather extends existing
    optimizers with decoupled weight decay. We explicitly define the two
    examples used in the above paper (SGDW and AdamW), but in general this can
    extend any OptimizerX class by using
        `ExtendedCls = extend_with_decoupled_weight_decay(OptimizerX)`.
    Weight decay can then be set when instantiating the optimizer:
        `optimizerX = ExtendedCls(weight_decay=0.001, learning_rate=0.001)`.
    In order for it to work, it must be the first class the Optimizer with
    weight decay inherits from, e.g.
    ```python
    class AdamW(DecoupledWeightDecayExtension, tf.keras.optimizers.Adam):
      def __init__(self, weight_decay, *args, **kwargs):
        super(AdamW, self).__init__(weight_decay, *args, **kwargs).
    ```
    Note: this extension decays weights BEFORE applying the update based
    on the gradient, i.e. this extension only has the desired behaviour for
    optimizers which do not depend on the value of'var' in the update step!
    Note: when applying a decay to the learning rate, be sure to manually apply
    the decay to the `weight_decay` as well. For example:
    ```python
    step = tf.Variable(0, trainable=False)
    schedule = tf.optimizers.schedules.PiecewiseConstantDecay(
        [10000, 15000], [1e-0, 1e-1, 1e-2])
    # lr and wd can be a function or a tensor
    lr = 1e-1 * schedule(step)
    wd = lambda: 1e-4 * schedule(step)
    # ...
    optimizer = tfa.optimizers.AdamW(learning_rate=lr, weight_decay=wd)
    ```
    """

    @typechecked
    def __init__(self, weight_decay: Union[FloatTensorLike, Callable], **kwargs):
        """Extension class that adds weight decay to an optimizer.
        Args:
            weight_decay: A `Tensor`, a floating point value, or a schedule
                that is a `tf.keras.optimizers.schedules.LearningRateSchedule`
                to decay the variable by, in the update step.
            **kwargs: Optional list or tuple or set of `Variable` objects to
                decay.
        """
        wd = kwargs.pop("weight_decay", weight_decay)
        super().__init__(**kwargs)
        self._decay_var_list = None  # is set in minimize or apply_gradients
        self._set_hyper("weight_decay", wd)

    def get_config(self):
        config = super().get_config()
        config.update(
            {"weight_decay": self._serialize_hyperparameter("weight_decay")})
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        # LR handling copied from optimizer_v2.OptimizerV2
        if "learning_rate" in config:
            if isinstance(config["learning_rate"], dict):
                config["learning_rate"] = tf.keras.optimizers.schedules.deserialize(
                    config["learning_rate"], custom_objects=custom_objects
                )

        if "weight_decay" in config:
            if isinstance(config["weight_decay"], dict):
                config["weight_decay"] = tf.keras.optimizers.schedules.deserialize(
                    config["weight_decay"], custom_objects=custom_objects
                )

        return cls(**config)

    def minimize(
        self, loss, var_list, grad_loss=None, name=None, decay_var_list=None, tape=None
    ):
        """Minimize `loss` by updating `var_list`.
        This method simply computes gradient using `tf.GradientTape` and calls
        `apply_gradients()`. If you want to process the gradient before
        applying then call `tf.GradientTape` and `apply_gradients()` explicitly
        instead of using this function.
        Args:
            loss: `Tensor` or callable. If a callable, `loss` should take no
                arguments and return the value to minimize. If a `Tensor`, the
                `tape` argument must be passed.
            var_list: list or tuple of `Variable` objects to update to
                minimize `loss`, or a callable returning the list or tuple of
                `Variable` objects. Use callable when the variable list would
                otherwise be incomplete before `minimize` since the variables
                are created at the first time `loss` is called.
            grad_loss: Optional. A `Tensor` holding the gradient computed for
                `loss`.
            decay_var_list: Optional list of variables to be decayed. Defaults
                to all variables in var_list.
            name: Optional name for the returned operation.
            tape: (Optional) `tf.GradientTape`. If `loss` is provided as a
                `Tensor`, the tape that computed the `loss` must be provided.
        Returns:
            An Operation that updates the variables in `var_list`.
        Raises:
            ValueError: If some of the variables are not `Variable` objects.
        """
        self._decay_var_list = (
            set([v.ref() for v in decay_var_list]) if decay_var_list else False
        )
        return super().minimize(
            loss, var_list=var_list, grad_loss=grad_loss, name=name, tape=tape
        )

    def apply_gradients(self, grads_and_vars, name=None, decay_var_list=None, **kwargs):
        """Apply gradients to variables.
        This is the second part of `minimize()`. It returns an `Operation` that
        applies gradients.
        Args:
            grads_and_vars: List of (gradient, variable) pairs.
            name: Optional name for the returned operation.  Default to the
                name passed to the `Optimizer` constructor.
            decay_var_list: Optional list of variables to be decayed. Defaults
                to all variables in var_list.
            **kwargs: Additional arguments to pass to the base optimizer's
                apply_gradient method, e.g., TF2.2 added an argument
                `experimental_aggregate_gradients`.
        Returns:
            An `Operation` that applies the specified gradients.
        Raises:
            TypeError: If `grads_and_vars` is malformed.
            ValueError: If none of the variables have gradients.
        """
        self._decay_var_list = (
            set([v.ref() for v in decay_var_list]) if decay_var_list else False
        )
        return super().apply_gradients(grads_and_vars, name=name, **kwargs)

    def _decay_weights_op(self, var, apply_state=None):
        if not self._decay_var_list or var.ref() in self._decay_var_list:
            var_device, var_dtype = var.device, var.dtype.base_dtype
            coefficients = (apply_state or {}).get(
                (var_device, var_dtype)
            ) or self._fallback_apply_state(var_device, var_dtype)

            return var.assign_sub(coefficients["wd_t"] * var, self._use_locking)
        return tf.no_op()

    def _decay_weights_sparse_op(self, var, indices, apply_state=None):
        if not self._decay_var_list or var.ref() in self._decay_var_list:
            var_device, var_dtype = var.device, var.dtype.base_dtype
            coefficients = (apply_state or {}).get(
                (var_device, var_dtype)
            ) or self._fallback_apply_state(var_device, var_dtype)

            update = -coefficients["wd_t"] * tf.gather(var, indices)
            return self._resource_scatter_add(var, indices, update)
        return tf.no_op()

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(DecoupledWeightDecayExtension, self)._prepare_local(
            var_device, var_dtype, apply_state
        )

        if "weight_decay" in self._hyper:
            wd_t = tf.identity(self._decayed_wd(var_dtype))
            apply_state[(var_device, var_dtype)]["wd_t"] = wd_t

    def _decayed_wd(self, var_dtype):
        wd_t = self._get_hyper("weight_decay", var_dtype)

        if isinstance(wd_t, tf.keras.optimizers.schedules.LearningRateSchedule):
            wd_t = tf.cast(wd_t(self.iterations), var_dtype)

        return wd_t

    # Here, we overwrite the apply functions that the base optimizer calls.
    # super().apply_x resolves to the apply_x function of the BaseOptimizer.

    def _resource_apply_dense(self, grad, var, apply_state=None):
        with tf.control_dependencies(
            [self._decay_weights_op(var, apply_state=apply_state)]
        ):
            return super()._resource_apply_dense(grad, var, apply_state=apply_state)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        decay_op = self._decay_weights_sparse_op(
            var, indices, apply_state=apply_state)
        with tf.control_dependencies([decay_op]):
            return super()._resource_apply_sparse(
                grad, var, indices, apply_state=apply_state
            )


class AdamW(DecoupledWeightDecayExtension, tf.keras.optimizers.Adam):
    """Optimizer that implements the Adam algorithm with weight decay.
    This is an implementation of the AdamW optimizer described in "Decoupled
    Weight Decay Regularization" by Loshch ilov & Hutter
    (https://arxiv.org/abs/1711.05101)
    ([pdf])(https://arxiv.org/pdf/1711.05101.pdf).
    It computes the update step of `tf.keras.optimizers.Adam` and additionally
    decays the variable. Note that this is different from adding L2
    regularization on the variables to the loss: it regularizes variables with
    large gradients more than L2 regularization would, which was shown to yield
    better training loss and generalization error in the paper above.
    For further information see the documentation of the Adam Optimizer.
    This optimizer can also be instantiated as
    ```python
    extend_with_decoupled_weight_decay(tf.keras.optimizers.Adam,
                                       weight_decay=weight_decay)
    ```
    Note: when applying a decay to the learning rate, be sure to manually apply
    the decay to the `weight_decay` as well. For example:
    ```python
    step = tf.Variable(0, trainable=False)
    schedule = tf.optimizers.schedules.PiecewiseConstantDecay(
        [10000, 15000], [1e-0, 1e-1, 1e-2])
    # lr and wd can be a function or a tensor
    lr = 1e-1 * schedule(step)
    wd = lambda: 1e-4 * schedule(step)
    # ...
    optimizer = tfa.optimizers.AdamW(learning_rate=lr, weight_decay=wd)
    ```
    """

    @typechecked
    def __init__(
        self,
        weight_decay: Union[FloatTensorLike, Callable],
        learning_rate: Union[FloatTensorLike, Callable] = 0.001,
        beta_1: Union[FloatTensorLike, Callable] = 0.9,
        beta_2: Union[FloatTensorLike, Callable] = 0.999,
        epsilon: FloatTensorLike = 1e-07,
        amsgrad: bool = False,
        name: str = "AdamW",
        **kwargs,
    ):
        """Construct a new AdamW optimizer.
        For further information see the documentation of the Adam Optimizer.
        Args:
            weight_decay: A Tensor or a floating point value. The weight decay.
            learning_rate: A Tensor or a floating point value. The learning
                rate.
            beta_1: A float value or a constant float tensor. The exponential
                decay rate for the 1st moment estimates.
            beta_2: A float value or a constant float tensor. The exponential
                decay rate for the 2nd moment estimates.
            epsilon: A small constant for numerical stability. This epsilon is
                "epsilon hat" in the Kingma and Ba paper (in the formula just
                before Section 2.1), not the epsilon in Algorithm 1 of the
                paper.
            amsgrad: boolean. Whether to apply AMSGrad variant of this
                algorithm from the paper "On the Convergence of Adam and
                beyond".
            name: Optional name for the operations created when applying
                gradients. Defaults to "AdamW".
            **kwargs: keyword arguments. Allowed to be {`clipnorm`,
                `clipvalue`, `lr`, `decay`}. `clipnorm` is clip gradients by
                norm; `clipvalue` is clip gradients by value, `decay` is
                included for backward compatibility to allow time inverse decay
                of learning rate. `lr` is included for backward compatibility,
                recommended to use `learning_rate` instead.
        """
        super().__init__(
            weight_decay,
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            amsgrad=amsgrad,
            name=name,
            **kwargs,
        )


class AveragedOptimizerWrapper(tf.keras.optimizers.Optimizer, metaclass=abc.ABCMeta):
    @typechecked
    def __init__(
        self, optimizer: Optimizer, name: str = "AverageOptimizer", **kwargs
    ):
        super().__init__(name, **kwargs)

        if isinstance(optimizer, str):
            optimizer = tf.keras.optimizers.get(optimizer)

        if not isinstance(optimizer, tf.keras.optimizers.Optimizer):
            raise TypeError(
                "optimizer is not an object of tf.keras.optimizers.Optimizer"
            )

        self._optimizer = optimizer
        self._track_trackable(self._optimizer, "awg_optimizer")

    def _create_slots(self, var_list):
        self._optimizer._create_slots(var_list=var_list)
        for var in var_list:
            self.add_slot(var, "average")

    def _create_hypers(self):
        self._optimizer._create_hypers()

    def _prepare_local(self, var_device, var_dtype, apply_state):
        return self._optimizer._prepare_local(var_device, var_dtype, apply_state)

    def apply_gradients(self, grads_and_vars, name=None, **kwargs):
        self._optimizer._iterations = self.iterations
        return super().apply_gradients(grads_and_vars, name, **kwargs)

    @abc.abstractmethod
    def average_op(self, var, average_var, local_apply_state):
        raise NotImplementedError

    def _apply_average_op(self, train_op, var, apply_state):
        apply_state = apply_state or {}
        local_apply_state = apply_state.get((var.device, var.dtype.base_dtype))
        if local_apply_state is None:
            local_apply_state = self._fallback_apply_state(
                var.device, var.dtype.base_dtype
            )
        average_var = self.get_slot(var, "average")
        return self.average_op(var, average_var, local_apply_state)

    def _resource_apply_dense(self, grad, var, apply_state=None):
        if "apply_state" in self._optimizer._dense_apply_args:
            train_op = self._optimizer._resource_apply_dense(
                grad, var, apply_state=apply_state
            )
        else:
            train_op = self._optimizer._resource_apply_dense(grad, var)
        average_op = self._apply_average_op(train_op, var, apply_state)
        return tf.group(train_op, average_op)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        if "apply_state" in self._optimizer._sparse_apply_args:
            train_op = self._optimizer._resource_apply_sparse(
                grad, var, indices, apply_state=apply_state
            )
        else:
            train_op = self._optimizer._resource_apply_sparse(
                grad, var, indices)
        average_op = self._apply_average_op(train_op, var, apply_state)
        return tf.group(train_op, average_op)

    def _resource_apply_sparse_duplicate_indices(
        self, grad, var, indices, apply_state=None
    ):
        if "apply_state" in self._optimizer._sparse_apply_args:
            train_op = self._optimizer._resource_apply_sparse_duplicate_indices(
                grad, var, indices, apply_state=apply_state
            )
        else:
            train_op = self._optimizer._resource_apply_sparse_duplicate_indices(
                grad, var, indices
            )
        average_op = self._apply_average_op(train_op, var, apply_state)
        return tf.group(train_op, average_op)

    def assign_average_vars(self, var_list):
        """Assign variables in var_list with their respective averages.
        Args:
            var_list: List of model variables to be assigned to their average.
        Returns:
            assign_op: The op corresponding to the assignment operation of
            variables to their average.
        Example:
        ```python
        model = tf.Sequential([...])
        opt = tfa.optimizers.SWA(
                tf.keras.optimizers.SGD(lr=2.0), 100, 10)
        model.compile(opt, ...)
        model.fit(x, y, ...)
        # Update the weights to their mean before saving
        opt.assign_average_vars(model.variables)
        model.save('model.h5')
        ```
        """
        assign_ops = []
        for var in var_list:
            try:
                assign_ops.append(
                    var.assign(
                        self.get_slot(var, "average"), use_locking=self._use_locking
                    )
                )
            except Exception as e:
                warnings.warn(
                    "Unable to assign average slot to {} : {}".format(var, e))
        return tf.group(assign_ops)

    def get_config(self):
        config = {
            "optimizer": tf.keras.optimizers.serialize(self._optimizer),
        }
        base_config = super().get_config()
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        optimizer = tf.keras.optimizers.deserialize(
            config.pop("optimizer"), custom_objects=custom_objects
        )
        return cls(optimizer, **config)

    @property
    def weights(self):
        return self._weights + self._optimizer.weights

    @property
    def lr(self):
        return self._optimizer._get_hyper("learning_rate")

    @lr.setter
    def lr(self, lr):
        self._optimizer._set_hyper("learning_rate", lr)  #

    @property
    def learning_rate(self):
        return self._optimizer._get_hyper("learning_rate")

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        self._optimizer._set_hyper("learning_rate", learning_rate)


class MovingAverage(AveragedOptimizerWrapper):
    """Optimizer that computes a moving average of the variables.
    Empirically it has been found that using the moving average of the trained
    parameters of a deep network is better than using its trained parameters
    directly. This optimizer allows you to compute this moving average and swap
    the variables at save time so that any code outside of the training loop
    will use by default the average values instead of the original ones.
    Example of usage:
    ```python
    opt = tf.keras.optimizers.SGD(learning_rate)
    opt = tfa.optimizers.MovingAverage(opt)
    ```
    """

    @typechecked
    def __init__(
        self,
        optimizer: Optimizer,
        average_decay: FloatTensorLike = 0.99,
        num_updates: Union[None, int, tf.Variable] = None,
        start_step: int = 0,
        dynamic_decay: bool = False,
        name: str = "MovingAverage",
        **kwargs,
    ):
        r"""Construct a new MovingAverage optimizer.
        Args:
            optimizer: str or `tf.keras.optimizers.Optimizer` that will be
                used to compute and apply gradients.
            average_decay: float. Decay to use to maintain the moving averages
                of trained variables.
            num_updates: Optional count of the number of updates applied to
                variables.
            start_step: int. What step to start the moving average.
            dynamic_decay: bool. Whether to change the decay based on the number
                of optimizer updates. Decay will start at 0.1 and gradually
                increase up to `average_decay` after each optimizer update.
            name: Optional name for the operations created when applying
                gradients. Defaults to "MovingAverage".
            **kwargs: keyword arguments. Allowed to be {`clipnorm`,
                `clipvalue`, `lr`, `decay`}. `clipnorm` is clip gradients by
                norm; `clipvalue` is clip gradients by value, `decay` is
                included for backward compatibility to allow time inverse
                decay of learning rate. `lr` is included for backward
                compatibility, recommended to use `learning_rate` instead.
        """
        super().__init__(optimizer, name, **kwargs)
        self._num_updates = num_updates
        if self._num_updates is not None:
            if isinstance(self._num_updates, tf.Variable):
                tf.debugging.assert_integer(
                    self._num_updates,
                    (
                        'type of argument "num_updates" must be '
                        "int; got {} instead".format(self._num_updates.dtype)
                    ),
                )
            num_updates = tf.cast(
                self._num_updates, tf.float32, name="num_updates")
            average_decay = tf.minimum(
                average_decay, (1.0 + num_updates) / (10.0 + num_updates)
            )

        self._set_hyper("average_decay", average_decay)
        self._start_step = start_step
        self._dynamic_decay = dynamic_decay

    @tf.function
    def _get_decay(self, step: tf.Tensor):
        average_decay = self._get_hyper("average_decay", tf.dtypes.float32)

        step = tf.cast(step, tf.float32)
        if step < self._start_step:
            return tf.constant(0.0, tf.float32)
        elif self._dynamic_decay:
            step_count = step - self._start_step
            return tf.minimum(average_decay, (1.0 + step_count) / (10.0 + step_count))
        else:
            return average_decay

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super()._prepare_local(var_device, var_dtype, apply_state)
        apply_state[(var_device, var_dtype)]["tfa_ma_decay"] = self._get_decay(
            self._optimizer.iterations
        )

    def average_op(self, var, average_var, local_apply_state):
        return tf.keras.backend.moving_average_update(
            average_var, var, local_apply_state["tfa_ma_decay"]
        )

    def get_config(self):
        config = {
            "average_decay": self._serialize_hyperparameter("average_decay"),
            "num_updates": self._num_updates,
            "start_step": self._start_step,
            "dynamic_decay": self._dynamic_decay,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def _create_slots(self, var_list):
        self._optimizer._create_slots(var_list=var_list)
        for var in var_list:
            self.add_slot(var, "average", var.read_value())

        self._average_weights = [self.get_slot(
            var, "average") for var in var_list]
        self._model_weights = var_list

    def shadow_copy(self, model_weights):
        """Creates shadow variables for the given model weights."""
        for var in model_weights:
            self.add_slot(var, "average", initializer="zeros")
        self._average_weights = [self.get_slot(
            var, "average") for var in model_weights]
        self._model_weights = model_weights

    @property
    def has_shadow_copy(self):
        """Whether this optimizer has created shadow variables."""
        return self._model_weights is not None

    def swap_weights(self):
        """Swap the average and moving weights.
        This is a convenience method to allow one to evaluate the averaged weights
        at test time. Loads the weights stored in `self._average_weights` into the model,
        keeping a copy of the original model weights. Swapping twice will return
        the original weights.
        """
        if tf.distribute.in_cross_replica_context():
            strategy = tf.distribute.get_strategy()
            return strategy.run(self._swap_weights, args=())
        else:
            raise ValueError(
                "Swapping weights must occur under a " "tf.distribute.Strategy"
            )

    @tf.function
    def _swap_weights(self):
        def fn_0(a, b):
            return a.assign_add(b, use_locking=self._use_locking)

        def fn_1(b, a):
            return b.assign(a - b, use_locking=self._use_locking)

        def fn_2(a, b):
            return a.assign_sub(b, use_locking=self._use_locking)

        def swap(strategy, a, b):
            """Swap `a` and `b` and mirror to all devices."""
            for a_element, b_element in zip(a, b):
                strategy.extended.update(
                    a_element, fn_0, args=(b_element,)
                )  # a = a + b
                strategy.extended.update(
                    b_element, fn_1, args=(a_element,)
                )  # b = a - b
                strategy.extended.update(
                    a_element, fn_2, args=(b_element,)
                )  # a = a - b

        ctx = tf.distribute.get_replica_context()
        return ctx.merge_call(swap, args=(self._average_weights, self._model_weights))
