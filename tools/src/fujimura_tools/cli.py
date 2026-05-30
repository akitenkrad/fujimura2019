"""fujimura-tools — unified CLI dispatcher.

    fujimura-tools visualize                  # time-series metrics + motive_mix + path diagram
    fujimura-tools visualize-sweep            # path-coefficient forest plot + SEM fit-index heatmap
    fujimura-tools show-experiment-settings   # print config / sweep_config / llm_meta
    fujimura-tools fit-sem                     # semopy: estimate the 4 path β̃ + fit indices
    fujimura-tools reproduce                   # Fig.1-style SEM path diagram + B1--B5 anchor reconciliation

Arguments after the subcommand are passed verbatim to that subcommand's argparse.
Add `--help` after a subcommand for its own help. The dispatcher assembly is
delegated to the shared helper `socsim_tools.cli.build_dispatcher`.
"""

from __future__ import annotations

from socsim_tools.cli import build_dispatcher

main = build_dispatcher(
    prog="fujimura-tools",
    description="Fujimura & Hino (2019) silence-and-voice — visualization, SEM fitting, reproduction",
    subcommands={
        "visualize": (
            "time-series metrics + motive_mix + estimated SEM path diagram",
            "fujimura_tools.visualize:main",
        ),
        "visualize-sweep": (
            "path-coefficient forest plot + SEM fit-index heatmap over the sweep",
            "fujimura_tools.visualize_sweep:main",
        ),
        "show-experiment-settings": (
            "print a results directory's settings (config / sweep_config / llm_meta)",
            "fujimura_tools.show_experiment_settings:main",
        ),
        "fit-sem": (
            "semopy: fit the ABM-induced SEM to agent_panel.csv, estimate the 4 path β̃ + CFI/GFI/RMSEA",
            "fujimura_tools.fit_sem:main",
        ),
        "reproduce": (
            "Fig.1-style SEM path diagram + the B1--B5 paper-anchor reconciliation report",
            "fujimura_tools.reproduce_paper:main",
        ),
    },
)


if __name__ == "__main__":
    main()
