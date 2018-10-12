package(default_visibility = ["//visibility:public"])

load("//tensorboard/defs:web.bzl", "tf_web_library")

licenses(["notice"])  # Apache 2.0

tf_web_library(
    name = "tf_agent_dashboard",
    srcs = [
        "tf-agent-dashboard.html",
        "tf-agent-video.html",
        "tf-agent-graph-card.html"
    ],
    path = "/tf-agent-dashboard",
    deps = [
        "//tensorboard/components/tf_backend",
        "//tensorboard/components/tf_card_heading",
        "//tensorboard/components/tf_categorization_utils",
        "//tensorboard/components/tf_color_scale",
        "//tensorboard/components/tf_dashboard_common",
        "//tensorboard/components/tf_imports:d3",
        "//tensorboard/components/tf_imports:lodash",
        "//tensorboard/components/tf_imports:polymer",
        "//tensorboard/components/tf_line_chart_data_loader",
        "//tensorboard/components/tf_runs_selector",
        "//tensorboard/components/tf_tensorboard:registry",
        "@org_polymer_paper_button",
        "@org_polymer_paper_dialog",
        "@org_polymer_paper_dropdown_menu",
        "@org_polymer_paper_icon_button",
        "@org_polymer_paper_item",
        "@org_polymer_paper_listbox",
        "@org_polymer_paper_menu_button",
        "@org_polymer_paper_menu",
        "@org_polymer_paper_radio_group",
        "@org_polymer_paper_slider",
        "@org_polymer_paper_spinner",
        "@org_polymer_paper_tooltip",
    ],
)
