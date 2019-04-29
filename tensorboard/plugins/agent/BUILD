# Description:
# TensorBoard plugin for viewing model data as a live video during training.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "agent",
    srcs = [
        "__init__.py",
        "agent.py",
    ],
    data = ["resources"],
    srcs_version = "PY2AND3",
    deps = [
        ":file_system_tools",
        ":im_util",
        ":shared_config",
        ":video_writing",
        ":visualizer",
        "//tensorboard:expect_numpy_installed",
        "//tensorboard:expect_tensorflow_installed",
    ],
)

py_library(
    name = "agent_plugin",
    srcs = ["agent_plugin.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":agent",
        ":file_system_tools",
        ":im_util",
        ":shared_config",
        "//tensorboard:expect_numpy_installed",
        "//tensorboard:expect_tensorflow_installed",
        "//tensorboard/util:encoder",
        "//tensorboard/util:tb_logging",
        "//tensorboard/backend:http_util",
        "//tensorboard/backend/event_processing:plugin_asset_util",
        "//tensorboard/plugins:base_plugin",
        "@org_pocoo_werkzeug",
    ],
)

py_binary(
    name = "agent_demo",
    srcs = ["agent_demo.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":agent",
        "//tensorboard:expect_tensorflow_installed",
    ],
)

py_library(
    name = "colormaps",
    srcs = ["colormaps.py"],
    licenses = ["unencumbered"],  # CC0
    srcs_version = "PY2AND3",
    deps = ["//tensorboard:expect_numpy_installed"],
)

py_library(
    name = "shared_config",
    srcs = ["shared_config.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "file_system_tools",
    srcs = ["file_system_tools.py"],
    data = ["resources"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorboard:expect_tensorflow_installed",
    ],
)

py_library(
    name = "im_util",
    srcs = ["im_util.py"],
    data = ["resources"],  # TODO(jart): Don't reference directory.
    srcs_version = "PY2AND3",
    deps = [
        ":colormaps",
        "//tensorboard:expect_numpy_installed",
        "//tensorboard:expect_tensorflow_installed",
        "//tensorboard/util:encoder",
        "//tensorboard/util:tb_logging",
    ],
)

py_library(
    name = "visualizer",
    srcs = ["visualizer.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":file_system_tools",
        ":im_util",
        ":shared_config",
        "//tensorboard:expect_numpy_installed",
        "//tensorboard:expect_tensorflow_installed",
    ],
)

py_library(
    name = "video_writing",
    srcs = ["video_writing.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":im_util",
        "//tensorboard/util:tb_logging",
    ],
)
