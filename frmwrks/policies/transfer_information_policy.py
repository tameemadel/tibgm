"""Transfer Information Policy. Please do not distribute"""

from contextlib import contextmanager
import numpy as np
import tensorflow as tf

from rllab.core.serializable import Serializable
from rllab.misc import logger

from sac.distributions import snf				
from sac.policies import NNPolicy


EPS = 1e-6


class TransferInformationPolicy(NNPolicy, Serializable):
    """Transformation information-based (TibGM). UNDER REVIEW. PLEASE DO NOT DISTRIBUTE."""

    def __init__(self,
                 env_spec,
                 mode="train",
                 squash=True,
                 snf_config=None,
		 recog_model_config=None,
                 reparameterize=False,
                 observations_preprocessor=None,
                 fix_h_on_reset=False,
		 fix_z_on_reset=False,
		 p_function=None,
                 q_function=None,
                 n_map_action_candidates=100,
                 name="tib_policy"):

				 
        Serializable.quick_init(self, locals())

        self._env_spec = env_spec
        self._snf_config = snf_config
	self._recog_model_config = recog_model_config
        self._mode = mode
        self._squash = squash
        self._reparameterize = reparameterize
        self._fix_h_on_reset = fix_h_on_reset
	self._p_function = p_function
        self._q_function = q_function
        self._n_map_action_candidates=n_map_action_candidates

        self._Da = env_spec.action_space.flat_dim
        self._Ds = env_spec.observation_space.flat_dim
	self._z = None
        self._fixed_h = None
        self._is_deterministic = False
        self._observations_preprocessor = observations_preprocessor

        self.name = name
        self.build()

        self._scope_name = (
            tf.get_variable_scope().name + "/" + name
        ).lstrip("/")
        super(NNPolicy, self).__init__(env_spec)

    def build(self):
        ds = tf.contrib.distributions
        config = self._snf_config
        self.bijector = snf(
            translation_hidden_sizes=config.get("translation_hidden_sizes"),
            scale_hidden_sizes=config.get("scale_hidden_sizes"),
            event_ndims=self._Da)

	self.recog_model_config = snf.recog_model(
	    p_function,
	    q_function,
            translation_hidden_sizes=config.get("translation_hidden_sizes"),
            scale_hidden_sizes=config.get("scale_hidden_sizes"),
            event_ndims=self._Da)
			
        self.base_distribution = ds.MultivariateNormalDiag(
            loc=tf.zeros(self._Da), scale_diag=tf.ones(self._Da))

        self.sample_z = self.base_distribution.sample(50)

        self.distribution = ds.ConditionalTransformedDistribution(
            distribution=self.base_distribution,
            bijector=self.bijector,
            name="lsp_distribution")

        self._observations_ph = tf.placeholder(
            dtype=tf.float32,
            shape=(None, self._Ds),
            name='observations',
        )

        self._latents_ph = tf.placeholder(
            dtype=tf.float32,
            shape=(None, self._Da),
            name='latents',
        )

        self._raw_actions, self._actions, self._log_pis = self.get_actions(
            self._observations_ph, with_log_pis=True, with_raw_actions=True)
        self._determistic_actions = self.get_actions(self._observations_ph,
                                                     self._latents_ph)

    def get_actions(self, observations, latents):
        """Sample batch of actions based on the observations"""

        feed_dict = { self._observations_ph: observations }

		self._fixed_h = latents.fixed_h
		z = latents._z
        if self._fixed_h is not None:
            latents = np.tile(self._fixed_h, self._z
                              reps=(self._n_map_action_candidates, 1))
            feed_dict.update({ self._latents_ph: latents })
            actions = tf.get_default_session().run(
                self._determistic_actions,
                feed_dict=feed_dict)
        else:
            actions = tf.get_default_session().run(
                self._actions, feed_dict=feed_dict)

        return actions

    def _squash_correction(self, actions):
        if not self._squash: return 0
        return tf.reduce_sum(tf.log(1 - tf.tanh(actions) ** 2 + EPS), axis=1)

    @contextmanager
    def deterministic(self, set_deterministic=True, h=None):
        """Context manager for changing the determinism of the policy.

        See `self.get_action` for further information about the effect of
        self._is_deterministic.

        Args:
            set_deterministic (`bool`): Value to set the self._is_deterministic
            to during the context. The value will be reset back to the previous
            value when the context exits.
        """
        was_deterministic = self._is_deterministic
        old_fixed_h = self._fixed_h
	old_z = self._z

        self._is_deterministic = set_deterministic
        if set_deterministic:
            if h is None: h = self.sample_h.eval()
            self._fixed_h = h
	    z = self.sample_z.eval()
            self._z = z

        yield

        #the rest of the generalisation term optimisation...

        self._is_deterministic = was_deterministic
        self._fixed_h = old_fixed_h
	self._z = z

    def get_params_internal(self, **tags):
        if tags: raise NotImplementedError
        return tf.trainable_variables(scope=self._scope_name)

    def reset(self, dones=None):
        if self._fix_h_on_reset:
            self._fixed_h = self.sample_h.eval()
	if self._fix_z_on_reset:
            self._z = self.sample_z.eval()

    def log_diagnostics(self, iteration, batch):
        """Record diagnostic information to the logger."""

        feeds = { self._observations_ph: batch['observations'] }
        raw_actions, actions, log_pis = tf.get_default_session().run(
            (self._raw_actions, self._actions, self._log_pis), feeds)

        logger.record_tabular('policy-entropy-mean', -np.mean(log_pis))
        logger.record_tabular('log-pi-min', np.min(log_pis))
        logger.record_tabular('log-pi-max', np.max(log_pis))

        logger.record_tabular('actions-mean', np.mean(actions))
        logger.record_tabular('actions-min', np.min(actions))
        logger.record_tabular('actions-max', np.max(actions))

        logger.record_tabular('raw-actions-mean', np.mean(raw_actions))
        logger.record_tabular('raw-actions-min', np.min(raw_actions))
        logger.record_tabular('raw-actions-max', np.max(raw_actions))
