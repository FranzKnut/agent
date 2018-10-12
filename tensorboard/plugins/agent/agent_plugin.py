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

import threading
import time

import base64
import os 
import numpy as np
import tensorflow as tf
from google.protobuf import message
from werkzeug import wrappers

from tensorboard import util
from tensorboard.backend import http_util
from tensorboard.backend.event_processing import plugin_asset_util as pau
from tensorboard.plugins import base_plugin
from tensorboard.plugins.agent import file_system_tools
from tensorboard.plugins.agent import im_util
from tensorboard.plugins.agent import shared_config

DEFAULT_INFO = [{
    'name': 'Waiting for data...',
}]

class AgentPlugin(base_plugin.TBPlugin):
    """
    TensorBoard plugin for interpreting reinforcement learning agents.
    """

    plugin_name = shared_config.PLUGIN_NAME

    def __init__(self, context):
        self._lock = threading.Lock()
        self._MULTIPLEXER = context.multiplexer
        self.PLUGIN_LOGDIR = pau.PluginDirectory(
            context.logdir, shared_config.PLUGIN_NAME)
        self.record_freq = 50
        self._config_file_lock = threading.Lock()
        self.current_paths = []
        self.current_episode = ""

    def get_plugin_apps(self):
        return {
            '/change-config': self._serve_change_config,
            '/ping': self._serve_ping,
            '/is-active': self._serve_is_active,
            '/episodes':self._serve_episodes,
            '/images': self._serve_images,
            '/image': self._serve_image,
            '/image-zip':self._serve_image_zip,
        }

    def is_active(self):
        folders = self._folder_list()
        return len(folders) > 0

    def is_config_writable(self):
        try:
            if not tf.gfile.Exists(self.PLUGIN_LOGDIR):
                tf.gfile.MakeDirs(self.PLUGIN_LOGDIR)
            config_filename = '{}/{}'.format(
                self.PLUGIN_LOGDIR, shared_config.CONFIG_FILENAME)
            with self._config_file_lock:
                file_system_tools.write_pickle(
                    file_system_tools.read_pickle(
                        config_filename, shared_config.DEFAULT_CONFIG),
                    config_filename)
            return True
        except tf.errors.PermissionDeniedError as e:
            tf.logging.warning(
                'Unable to write Agent config, controls will be disabled: %s', e)
            return False

    def _folder_list(self):
        d=self.PLUGIN_LOGDIR
        folders = list(filter(lambda x: os.path.isdir(os.path.join(d, x)), os.listdir(d)))
        return folders


    @wrappers.Request.application
    def _serve_is_active(self, request):
        is_active = self.is_active()
        # If the plugin isn't active, don't check if the configuration is writable
        # since that will leave traces on disk; instead return True (the default).
        is_config_writable = self.is_config_writable() if is_active else True
        response = {
            'is_active': is_active,
            'is_config_writable': is_config_writable,
        }
        return http_util.Respond(request, response, 'application/json')

    @wrappers.Request.application
    def _serve_episodes(self,request):
        folders = sorted(self._folder_list(), reverse=True)
        return http_util.Respond(request, {"folders": folders}, 'application/json')
    
    @wrappers.Request.application
    def _serve_images(self, request):
        selected_episode = request.form["selected_episode"]
        view_mode = request.form["view_mode"]

        directory = '{}/{}/{}'.format(self.PLUGIN_LOGDIR, selected_episode, view_mode)
        base64_images = []

        for folder, subs, files in os.walk(directory):
            for filename in files:
                path = os.path.abspath(os.path.join(folder, filename))
                with open(path, "rb") as image_file:
                   base64_images.append(base64.b64encode(image_file.read()))


        return http_util.Respond(request, {"base64_images":base64_images}, 'application/json')

    @wrappers.Request.application
    def _serve_change_config(self, request):
        print("Serve change config")
        config = {}

        for key, value in request.form.items():
            try:
                config[key] = int(value)
            except ValueError:
                if value == 'false':
                    config[key] = False
                elif value == 'true':
                    config[key] = True
                else:
                    config[key] = value

        self.FPS = config['FPS']
        self.record_freq = config['record_freq']

        with self._config_file_lock:
            file_system_tools.write_pickle(
                config,
                '{}/{}'.format(self.PLUGIN_LOGDIR, shared_config.CONFIG_FILENAME))
        return http_util.Respond(request, {'config': config}, 'application/json')

    @wrappers.Request.application
    def _serve_ping(self, request):  # pylint: disable=unused-argument
        return http_util.Respond(request, {'status': 'alive'}, 'application/json')
