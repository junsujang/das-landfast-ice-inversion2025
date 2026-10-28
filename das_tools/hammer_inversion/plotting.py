import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PARAM_LABELS = {
    "E": "Young's modulus, $E$ [GPa]",
    "nu": "Poisson's ratio, $\\nu$",
    "h_i": "Ice thickness, $h_i$ [m]",
    "h_w": "Water depth, $h_w$ [m]",
}


def free_params(result):
    """Parameters the grid actually searched; a fixed one (e.g. nu held at 0.3) has one value."""
    return [p for p,axis in result.parameter_axes.items() if len(axis)>1]


def make_event_summary(event_results,ci_level=0.95,ci_method="equal_tailed"):
    """Per-event posterior summary.

    Adds, for every parameter ``p``, the credible-interval bounds
    ``p_ci_lo``/``p_ci_hi`` at ``ci_level`` (default 95%) alongside the MAP,
    posterior mean and standard deviation. ``ci_method`` is passed through to
    :meth:`PosteriorResult.marginal_credible_interval` ("equal_tailed" or "hpd").
    """
    rows=[]
    for event_id,result in sorted(event_results.items(),key=lambda x:x[0]):
        row={"event_id":event_id}
        for p in free_params(result):
            row[f"{p}_MAP"]=result.map_params[p]
            row[f"{p}_mean"]=result.marginal_mean(p)
            row[f"{p}_std"]=result.marginal_std(p)
            lo,hi=result.marginal_credible_interval(p,level=ci_level,method=ci_method)
            row[f"{p}_ci_lo"]=lo
            row[f"{p}_ci_hi"]=hi
        rows.append(row)
    df=pd.DataFrame(rows)
    df.attrs["ci_level"]=ci_level
    df.attrs["ci_method"]=ci_method
    return df


def plot_marginals(result,figsize=(12,4),ci_level=0.90,ci_method="equal_tailed",show_ci=True):
    """Plot 1-D marginals with the posterior mean and a shaded credible interval."""
    params=free_params(result)
    fig,axes=plt.subplots(1,len(params),figsize=figsize)
    if len(params)==1:
        axes=[axes]

    for ax,p in zip(axes,params):
        label=PARAM_LABELS.get(p,p)
        x=result.parameter_axes[p]
        if p == "E":
            x = x/1e9
            ax.set_xlim(0.0,10.0)
            ax.set_xticks(np.arange(1.0,10.5,2.0))
        if p == "h_i" and x.max() <= 1.25:  # PISCES-25 thickness range
            ax.set_xticks(np.arange(0.8,1.25,0.1))
        pdf=result.marginals[p]
        ax.plot(x,pdf)
        # ax.axvline(result.map_params[p],ls="--",label="MAP")
        mean=result.marginal_mean(p)
        std=result.marginal_std(p)
        lo,hi=result.marginal_credible_interval(p,level=ci_level,method=ci_method)
        print(f"{p}: mean={mean:.6g}, std={std:.6g}, "
              f"{ci_level:.0%} CI=[{lo:.6g}, {hi:.6g}] ({ci_method})")

        scale=1e9 if p == "E" else 1.0
        ax.axvline(mean/scale,ls=":",label="Posterior mean",linewidth=2,color='k')
        if show_ci:
            ax.axvspan(lo/scale,hi/scale,alpha=.15,color='k',
                       label=f"{ci_level:.0%} CI")
        ax.set_xlabel(label)
        if p == params[0]:
            ax.set_ylabel("Posterior probability");
        if p == params[0]:
            ax.legend(loc="upper left")
        ax.grid(True)
    # fig.tight_layout()
    return fig,axes


def _select_events(s,ids):
    """Rows of the summary frame for ``ids``.

    Values found in ``s.event_id`` select by event id; any remaining integers
    are treated as positional row indices (the frame is sorted by event id).
    """
    known=set(s.event_id)
    by_id=[i for i in ids if i in known]
    by_pos=[i for i in ids if i not in known]
    if by_pos and not all(isinstance(i,(int,np.integer)) for i in by_pos):
        raise KeyError(f"unknown event ids: {by_pos}")
    sel=s[s.event_id.isin(by_id)]
    if by_pos:
        sel=pd.concat([sel,s.iloc[list(by_pos)]]).drop_duplicates(subset="event_id")
    return sel.sort_values("event_id")


def plot_event_estimates(event_results,global_result=None,figsize=(12,10),
                         ci_level=0.95,ci_method="equal_tailed",show_ci=True,
                         param_events=None,ylims=None):
    """Per-event posterior means with 1-sigma bars and, if ``show_ci``, credible-interval bars.

    ``param_events`` optionally restricts individual panels to a subset of
    events, e.g. ``{"nu": [2, 4, 14, 15]}`` to show only those four events in
    the Poisson's ratio panel. Entries are event ids when they appear in the
    summary frame, otherwise positional row indices.

    ``ylims`` optionally sets per-panel y-limits in plotted units (GPa for
    ``E``), e.g. ``{"E": (4, 7)}``.
    """
    s = make_event_summary(event_results,ci_level=ci_level,ci_method=ci_method);
    params = free_params(next(iter(event_results.values())))
    fig,axes=plt.subplots(len(params),1,figsize=figsize,sharex=True,layout=None)
    fig.set_layout_engine(None)
    if len(params)==1:
        axes=[axes]
    specs = [(f"{p}_mean",f"{p}_std",p,PARAM_LABELS.get(p,p)) for p in params]

    param_events=param_events or {}
    ylims=ylims or {}

    for ax,(mc,sc,p,ylabel) in zip(axes,specs):
        sp=_select_events(s,param_events[p]) if p in param_events else s
        scale=1e9 if p == "E" else 1.0  # plot E in GPa
        if show_ci:
            ci_err=np.vstack([sp[mc]-sp[f"{p}_ci_lo"],sp[f"{p}_ci_hi"]-sp[mc]])/scale
            ax.errorbar(sp.event_id,sp[mc]/scale,yerr=ci_err,fmt="o",capsize=5,
                        elinewidth=1,color="C0",alpha=1,
                        label=f"Event {ci_level:.0%} CI")

        if global_result is not None:
            gm=global_result.marginal_mean(p)
            ax.axhline(gm/scale,ls="--",label="Global posterior mean")

        ax.set_ylabel(ylabel);ax.grid(alpha=.2);
        if p in ylims:
            ax.set_ylim(*ylims[p])
        if mc == specs[0][0]:
            ax.legend()
        
    axes[-1].set_xlabel("Event")
    # fig.tight_layout()
    return fig,axes


def plot_2d_marginal(result,parameter_x,parameter_y,xlabel=None,ylabel=None):
    names=list(result.parameter_axes);ix=names.index(parameter_x);iy=names.index(parameter_y)
    if ix==iy:raise ValueError("Choose two different parameters.")
    remaining=({0,1,2}-{ix,iy}).pop()
    p2=result.posterior.sum(axis=remaining)
    # After summing, surviving axes remain in original order. Reorder to y,x for contourf.
    surviving=[i for i in range(3) if i!=remaining]
    pos_x=surviving.index(ix);pos_y=surviving.index(iy)
    z=np.moveaxis(p2,(pos_y,pos_x),(0,1))
    x=result.parameter_axes[parameter_x];y=result.parameter_axes[parameter_y]

    fig,ax=plt.subplots(figsize=(6,5))
    im = ax.contourf(x,y,z,levels=20)
    ax.set_xlabel(xlabel or parameter_x)
    ax.set_ylabel(ylabel or parameter_y)
    ax.set_title(f"{result.label}: {parameter_y} vs {parameter_x}")
    fig.colorbar(im, ax=ax)
    # fig.tight_layout()
    return fig,ax
