"""
Visualization module for simulation results.

Generates charts: price history with event markers, wealth timeline,
and final rankings bar chart.
"""

from typing import Any, Dict, List, Optional


def _check_matplotlib():
    """Check if matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        return True
    except ImportError:
        return False


def plot_price_history(
    price_history: List[float],
    event_history: Optional[List[Dict[str, Any]]] = None,
    output_path: Optional[str] = None,
    title: str = "Price History",
) -> Optional[Any]:
    """
    Plot price history with optional event markers.

    Parameters
    ----------
    price_history : list of float
        Historical prices.
    event_history : list of dict, optional
        List of events with 'step', 'name', 'price_impact'.
    output_path : str, optional
        Path to save the figure. If None, returns the figure.
    title : str
        Chart title.

    Returns
    -------
    figure or None
        Returns figure if output_path is None, otherwise saves and returns None.
    """
    if not _check_matplotlib():
        print("Warning: matplotlib not installed. Skipping visualization.")
        return None

    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')

    fig, ax = plt.subplots(figsize=(12, 6))

    steps = list(range(len(price_history)))
    ax.plot(steps, price_history, 'b-', linewidth=1.5, label='Price')
    ax.fill_between(steps, price_history, alpha=0.1)

    # Mark events
    if event_history:
        for event in event_history:
            step = event.get("step", 0)
            if 0 <= step < len(price_history):
                impact = event.get("price_impact", 0)
                color = 'green' if impact >= 0 else 'red'
                marker = '^' if impact >= 0 else 'v'
                ax.scatter(
                    [step], [price_history[step]],
                    c=color, marker=marker, s=100, zorder=5,
                    label=event.get("name", "event")[:15]
                )

    ax.set_xlabel("Step")
    ax.set_ylabel("Price ($)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Price chart saved to: {output_path}")
        return None
    else:
        return fig


def plot_wealth_timeline(
    state_history: List[Dict[str, Any]],
    agent_ids: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    title: str = "Agent Wealth Timeline",
) -> Optional[Any]:
    """
    Plot wealth over time for all agents.

    Parameters
    ----------
    state_history : list of dict
        State snapshots from simulation.
    agent_ids : list of str, optional
        Specific agents to plot. If None, plots all.
    output_path : str, optional
        Path to save the figure.
    title : str
        Chart title.

    Returns
    -------
    figure or None
    """
    if not _check_matplotlib():
        print("Warning: matplotlib not installed. Skipping visualization.")
        return None

    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')

    if not state_history:
        return None

    # Extract agent IDs from first state
    if agent_ids is None:
        first_agents = state_history[0].get("agents", {})
        agent_ids = list(first_agents.keys())

    fig, ax = plt.subplots(figsize=(12, 6))

    for agent_id in agent_ids:
        wealth_history = []
        for state in state_history:
            agents_data = state.get("agents", {})
            if agent_id in agents_data:
                wealth_history.append(agents_data[agent_id].get("wealth", 0))
            else:
                wealth_history.append(0)

        ax.plot(
            range(len(wealth_history)),
            wealth_history,
            linewidth=1.5,
            label=agent_id[:20],
        )

    ax.set_xlabel("Step")
    ax.set_ylabel("Wealth ($)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize='small', ncol=2)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Wealth timeline saved to: {output_path}")
        return None
    else:
        return fig


def plot_final_rankings(
    final_state: Dict[str, Any],
    initial_price: float,
    output_path: Optional[str] = None,
    title: str = "Final Agent Rankings",
    top_n: int = 15,
    initial_wealths: Optional[Dict[str, float]] = None,
) -> Optional[Any]:
    """
    Plot final wealth rankings as horizontal bar chart.

    Parameters
    ----------
    final_state : dict
        Final state snapshot.
    initial_price : float
        Initial price for calculating returns.
    output_path : str, optional
        Path to save the figure.
    title : str
        Chart title.
    top_n : int
        Number of top agents to display.

    Returns
    -------
    figure or None
    """
    if not _check_matplotlib():
        print("Warning: matplotlib not installed. Skipping visualization.")
        return None

    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')

    agents_data = final_state.get("agents", {})
    if not agents_data:
        return None

    # Calculate wealth and sort
    rankings = []
    for agent_id, data in agents_data.items():
        wealth = data.get("wealth", 0)
        # Use the exact initial wealth captured before trading when provided;
        # fall back to an estimate from final holdings otherwise.
        initial_wealth = (initial_wealths or {}).get(agent_id)
        if initial_wealth is None:
            cash = data.get("cash", 0)
            holdings = data.get("holdings", 0)
            initial_wealth = cash + holdings * initial_price if holdings > 0 else cash
        ret = (wealth / initial_wealth - 1) * 100 if initial_wealth > 0 else 0
        rankings.append({
            "id": agent_id,
            "wealth": wealth,
            "return": ret,
        })

    rankings.sort(key=lambda x: x["wealth"], reverse=True)
    rankings = rankings[:top_n]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))

    names = [r["id"][:20] for r in rankings]
    wealths = [r["wealth"] for r in rankings]
    returns = [r["return"] for r in rankings]

    colors = ['green' if r >= 0 else 'red' for r in returns]

    bars = ax.barh(range(len(names)), wealths, color=colors, alpha=0.7)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Final Wealth ($)")
    ax.set_title(title)

    # Add return labels
    for i, (bar, ret) in enumerate(zip(bars, returns)):
        ax.text(
            bar.get_width() + 100,
            bar.get_y() + bar.get_height() / 2,
            f"{ret:+.1f}%",
            va='center',
            fontsize='small',
        )

    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Rankings chart saved to: {output_path}")
        return None
    else:
        return fig


def generate_all_charts(
    price_history: List[float],
    state_history: List[Dict[str, Any]],
    event_history: Optional[List[Dict[str, Any]]] = None,
    initial_price: float = 100.0,
    output_dir: str = ".",
    initial_wealths: Optional[Dict[str, float]] = None,
) -> Dict[str, Optional[Any]]:
    """
    Generate all visualization charts.

    Parameters
    ----------
    price_history : list of float
    state_history : list of dict
    event_history : list of dict, optional
    initial_price : float
    output_dir : str
        Directory to save charts.

    Returns
    -------
    figures : dict
        Dictionary of figure objects (None if saved to file).
    """
    import os

    return {
        "price": plot_price_history(
            price_history,
            event_history,
            output_path=os.path.join(output_dir, "price_history.png"),
        ),
        "wealth": plot_wealth_timeline(
            state_history,
            output_path=os.path.join(output_dir, "wealth_timeline.png"),
        ),
        "rankings": plot_final_rankings(
            state_history[-1] if state_history else {},
            initial_price,
            output_path=os.path.join(output_dir, "final_rankings.png"),
            initial_wealths=initial_wealths,
        ),
    }