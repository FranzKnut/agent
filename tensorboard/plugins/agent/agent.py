# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
from __future__ import division
from __future__ import print_function

import os
import time

import numpy as np
import tensorflow as tf
import json
import io
try:
    to_unicode = unicode
except NameError:
    to_unicode = str


from tensorboard.plugins.agent import im_util
from tensorboard.plugins.agent.file_system_tools import read_pickle,\
  write_pickle, write_file
from tensorboard.plugins.agent.shared_config import PLUGIN_NAME, TAG_NAME,\
   DEFAULT_CONFIG, CONFIG_FILENAME
from tensorboard.plugins.agent import video_writing
from tensorboard.plugins.agent.visualizer import Visualizer


class Agent(object):

  def __init__(self, logdir):
    self.PLUGIN_LOGDIR = logdir + '/plugins/' + PLUGIN_NAME
    self.LOG_DIR = None

    self.is_recording = True
    self.live_prefix = "_LIVE__"

    self.video_writer = None
    self.frame_placeholder = tf.placeholder(tf.uint8, [None, None, None])

    self.last_image_shape = []
    self.config_last_modified_time = -1
    self.previous_config = dict(DEFAULT_CONFIG)
    self.rewards = []
    self.actions = []
    self.episode_count = 0
    self.start_time = round(time.time())

    if not tf.gfile.Exists(self.PLUGIN_LOGDIR + '/config.pkl'):
      tf.gfile.MakeDirs(self.PLUGIN_LOGDIR)
      write_pickle(DEFAULT_CONFIG, '{}/{}'.format(self.PLUGIN_LOGDIR,
                                                  CONFIG_FILENAME))

    self.visualizer = Visualizer(self.PLUGIN_LOGDIR)


  def _get_config(self):
    '''Reads the config file from disk or creates a new one.'''
    filename = '{}/{}'.format(self.PLUGIN_LOGDIR, CONFIG_FILENAME)
    modified_time = os.path.getmtime(filename)

    if modified_time != self.config_last_modified_time:
      config = read_pickle(filename, default=self.previous_config)
      self.previous_config = config
    else:
      config = self.previous_config

    self.config_last_modified_time = modified_time
    return config


  def _get_final_image(self, session, config, frame=None, arrays=None):
    if config['values'] == 'frames':
      if frame is None:
        final_image = im_util.get_image_relative_to_script('frame-missing.png')
      else:
        frame = frame() if callable(frame) else frame
        final_image = im_util.scale_image_for_display(frame)

    if len(final_image.shape) == 2:
      # Map grayscale images to 3D tensors.
      final_image = np.expand_dims(final_image, -1)

    return final_image


  def _update_frame(self, session, frame, config):
    final_image = self._get_final_image(session, config, frame)
    self.last_image_shape = final_image.shape
    return final_image


  def _update_recording(self, frame, config):
    '''Adds a frame to the current video output.'''
    # pylint: disable=redefined-variable-type
    should_record = config['is_recording']

    if should_record:
      if not self.is_recording:
        self.is_recording = True
        tf.logging.info(
            'Starting recording using %s',
            self.video_writer.current_output().name())
      self.video_writer.write_frame(frame)
    elif self.is_recording:
      self.is_recording = False
      self.video_writer.finish()
      tf.logging.info('Finished recording')


  def update(self, session, env_name="env", tag="", frame=None, action=-1, reward=0.0, done=False):
    '''Updates Agent with information from a single step of the environment
    '''

    current_episode_count = self.episode_count

    if done:
      self.episode_count += 1

    new_config = self._get_config()
    record_freq = new_config['record_freq']

    if current_episode_count % record_freq != 0 or record_freq == 0:
      return

    if self.video_writer is None:
      self._start_episode(env_name.strip(), tag.strip())

    final_image = self._update_frame(session, frame, new_config)
    self._update_recording(final_image, new_config)

    self.actions.append(np.asscalar(action))
    self.rewards.append(np.asscalar(reward))

    if done:
      self._finish_episode(current_episode_count)

  def _start_episode(self, env_name, tag):
      # Directory
      d=self.PLUGIN_LOGDIR
      tagString = '' if tag == ''  else  '_{}'.format(tag)
      self.LOG_DIR = '{}/{}{}{}-{}-ep{}'.format(d, self.live_prefix, env_name, tag, self.start_time, str(self.episode_count).zfill(6) )

      # TODO: Create video writer for saliency
      self.video_writer = video_writing.VideoWriter(
        self.LOG_DIR + "/renders",
        outputs=[video_writing.PNGVideoOutput])


  def _finish_episode(self, current_episode_count):
        self.is_recording = False
        if self.video_writer is not None:
          self.video_writer.finish()
          self.video_writer = None

          completed_dir = self.LOG_DIR.replace(self.live_prefix, '')

          data = {"rewards":self.rewards, 
                  "actions":self.actions, 
                  "end_time":round(time.time()),
                  "cumulative_reward":sum(self.rewards),
                  "frame_count": len(self.rewards),
                  "episode_count": current_episode_count,
                  "name": completed_dir.replace(self.PLUGIN_LOGDIR, '')
                   }

          with io.open(self.LOG_DIR +'/metadata.json', 'w', encoding='utf8') as outfile:
            str_ = json.dumps(data,
                      indent=4, sort_keys=True,
                      separators=(',', ': '), ensure_ascii=False)
            outfile.write(to_unicode(str_))

          completed_dir = self.LOG_DIR.replace(self.live_prefix, '')
          os.rename(self.LOG_DIR, completed_dir)

        self.rewards = []
        self.actions = []
        self.LOG_DIR = None
        tf.logging.info('Finished recording episode')

  ##############################################################################

  @staticmethod
  def gradient_helper(optimizer, loss, var_list=None):
    '''A helper to get the gradients out at each step.

    Args:
      optimizer: the optimizer op.
      loss: the op that computes your loss value.

    Returns: the gradient tensors and the train_step op.
    '''
    if var_list is None:
      var_list = tf.trainable_variables()

    grads_and_vars = optimizer.compute_gradients(loss, var_list=var_list)
    grads = [pair[0] for pair in grads_and_vars]

    return grads, optimizer.apply_gradients(grads_and_vars)


class AgentHook(tf.train.SessionRunHook):
  """SessionRunHook implementation that runs Agent every step.

  Convenient when using tf.train.MonitoredSession:
  ```python
  agent_hook = AgentHook(LOG_DIRECTORY)
  with MonitoredSession(..., hooks=[agent_hook]) as sess:
    sess.run(train_op)
  ```
  """
  def __init__(self, logdir):
    """Creates new Hook instance

    Args:
      logdir: Directory where Agent should write data.
    """
    self._logdir = logdir
    self.agent = None

  def begin(self):
    self.agent = Agent(self._logdir)

  def after_run(self, run_context, unused_run_values):
    self.agent.update(run_context.session)
