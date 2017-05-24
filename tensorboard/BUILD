# Description:
# TensorBoard, a dashboard for investigating TensorFlow

package(default_visibility = ["//third_party/tensorflow:internal"])

licenses(["notice"])  # Apache 2.0

py_binary(
    name = "tensorboard",
    srcs = ["tensorboard.py"],
    data = [":assets"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorboard/backend:application",
        "//tensorboard/backend/event_processing:event_file_inspector",
        "//tensorboard/plugins/projector:projector_plugin",
        "//tensorboard/plugins/text:text_plugin",
        "//third_party/tensorflow/python:platform",
        "@org_pocoo_werkzeug//:werkzeug",
    ],
)

filegroup(
    name = "assets",
    srcs = [
        "TAG",
        "//tensorboard/components:index.html",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**"],
        exclude = [
            "METADATA",
            "OWNERS",
            "tensorboard.google.bzl",
        ],
    ),
    tags = ["notsan"],
)