# import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.figure import Figure
import pandas as pd
from scipy.interpolate import interp1d
from pathlib import Path
from .utilities import rt_quasi_deuteron
from .presets import *

# overall styling for scatter error bar plots
SCATTER_SETTING = {'s':10, 'edgecolors':'gray', 'linewidth':0.5, 'alpha':1}

ERRORBAR_SETTING = {'markersize':0, 'capsize':0, 'lw':1, 'fmt':'D', 'elinewidth':1, 'alpha':0.5}

EXP_PLOT_LIST = ['Yamaguchi', 'Barreau', 'Jourdan', 'Goldemberg', 
                #  'Baran', 'Sheren', 
                #  'Buki',
                 'Photo-production']

THEORY_PLOT_LIST = ['SuSAv2', 'ED-RMF', 'STA-QMC', 'CFG']

MC_PLOT_LIST = ['NuWro-SF', 'NuWro-SF-FSI', 'ACHILLES']
SCATTER_STYLES = {
    'Yamaguchi':{'color':'cyan', 's':10, 'marker':'o', 'edgecolors':'gray', 'linewidth':0.3},
    'Photo-production':{'s':30,'color':'lime','marker':'^','edgecolors':'gray','linewidth':0.5},
    'Goldemberg':{'color':'blue','marker':'d','s':15, 'edgecolors':'gray','linewidth':0.5},
    'Jourdan':{'color':'limegreen','marker':'v','s':15, 'edgecolors':'gray','linewidth':0.5},
    'Barreau':{'color':'yellow', 'marker':'H','s':15, 'edgecolors':'gray','linewidth':0.5},
    'Buki':{'color':'blue','marker':'P','s':15, 'edgecolors':'gray','linewidth':0.5},
    'Sheren':{'color':'deepskyblue','s':15, 'edgecolors':'gray','linewidth':0.5},
    'Baran':{'color':'orange','s':15, 'edgecolors':'gray','linewidth':0.5}
}

ERRORBAR_STYLES = {
    'Goldemberg':{'color':'skyblue','ecolor':'blue','markersize':0, 'capsize':0, 'lw':1, 'fmt':'D', 'elinewidth':1, 'alpha':0.5},
    'Yamaguchi':{'color':'gray','markersize':0, 'capsize':0, 'lw':0.5, 'elinewidth':1, 'alpha':0.5},
    'Jourdan':{'color':'limegreen','markersize':0, 'capsize':0, 'lw':1, 'fmt':'D', 'elinewidth':1, 'alpha':0.5},
    'Barreau':{'color':'black','markersize':0, 'capsize':0, 'lw':1, 'fmt':'D', 'elinewidth':1, 'alpha':0.5},
    'Buki':{'color':'blue','markersize':0, 'capsize':0, 'lw':1, 'fmt':'D', 'elinewidth':1, 'alpha':0.5},
    'Sheren':{'ecolor':'deepskyblue','markersize':0, 'capsize':0, 'lw':1, 'fmt':'D', 'elinewidth':1, 'alpha':0.5},
    'Baran':{'ecolor':'gray','markersize':0, 'capsize':0, 'lw':1, 'fmt':'D', 'elinewidth':1, 'alpha':0.5},
    'Photo-production':{}
}

LINE_STYLES = {
    'GFMC':{'color':'violet', 'linestyle':'solid','lw':2,},
    'ED-RMF':{'color':'cornflowerblue', 'linestyle':'solid','lw':3},
    'CFG':{'color':'brown','linestyle':'-','lw':2},
    'STA-QMC':{'color':'lime', 'linestyle':'solid','lw':2},
    'SuSAv2':{},
    'NuWro-SF':{'color':'violet','lw':2,'linestyle':'-'},
    'NuWro-SF-FSI':{'color':'green','lw':2,'linestyle':'-'},
    'ACHILLES':{'color':'blue','linestyle':'dotted','lw':2.5},
}

RLRT_QVPLOT_HEIGHTS = {
    0.1:[50.0,12],
    0.148:[90.0,27],
    0.167:[80.0,30],
    0.205:[80.0,35],
    0.24:[60.0,41],
    0.3:[50.0,44],
    0.38:[30.0,35],
    0.475:[17.5,45],
    0.57:[10.0,35],
    0.649:[10.0,40],
    0.756:[10.0,42],
    0.991:[10.0,25],
    1.619:[100.0,45],
    1.921:[120.0,35],
    2.213:[80.0,35],
    2.5:[50.0,26],
    2.783:[40.0,15],
    3.5:[12.5,12],
}
    # RL_plot_heights = np.array([50,90,80,80,60,
    #     50,30, 17.5, 10,10, 
    #     10, 10,100,120,80,
    #     50,40,12.5])
    # RT_plot_heights = np.array([12,27,30,35,41,
    #     44,35,45,35,40,
    #     42,25,45,35,35,
    #     26,15,12])

def plot_response_qvbin(df_this_analysis : pd.DataFrame, qvcenters : list[float] = [0.3, 0.38, 0.57], figsize_per_row : tuple[float, float] = (11, 2),
        sharex : bool = False, figshow : bool = False, christy_bodek_fit : bool = True,
        exp_plot_list : list[str] = EXP_PLOT_LIST, mc_plot_list : list[str] = MC_PLOT_LIST) -> Figure:


    fig, axs = plt.subplots(nrows=len(qvcenters), ncols=2, figsize=(figsize_per_row[0], figsize_per_row[1] * len(qvcenters) + 1.5), dpi=200, sharex=sharex, squeeze=False)
    scifmt = ScalarFormatter(useMathText=True)
    scifmt.set_scientific(True)
    scifmt.set_powerlimits((0, 0))
    handles = []
    labels = []

    repo_path = Path(__file__).resolve().parent.parent
    plot_xlsx = repo_path/'Carbon/one_sheet_to_rule_them_all.xlsx'
    sheet_CBfit = pd.read_excel(plot_xlsx, sheet_name='CBfit_qvbin')
    sheet_mc_rl = pd.read_excel(plot_xlsx, sheet_name='mc_rl_qvbin')
    sheet_mc_rt = pd.read_excel(plot_xlsx, sheet_name='mc_rt_qvbin')
    sheet_exp_rl = pd.read_excel(plot_xlsx, sheet_name='exp_rl_qvbin')
    sheet_exp_rt = pd.read_excel(plot_xlsx, sheet_name='exp_rt_qvbin')
    sheet_theory_rl = pd.read_excel(plot_xlsx, sheet_name='theory_rl_qvbin')
    sheet_theory_rt = pd.read_excel(plot_xlsx, sheet_name='theory_rt_qvbin')

    for i, qvcenter in enumerate(qvcenters):
        if qvcenter not in QVCENTERS:
            print(f'q={qvcenter} not in spreadsheet')
            continue

        # Christy Bodek Fit
        # 25 Sep 25: shift inelastic to left by 18 MeV; July 18 RT quasi deuteron added
        ChristyBodekFit = sheet_CBfit.loc[sheet_CBfit['qv']==qvcenter].copy()
        # print('len CBfit',len(ChristyBodekFit))
        f_RTIE = interp1d(ChristyBodekFit['nu'], ChristyBodekFit['rtie'], kind="linear", bounds_error=False, fill_value=0.0)
        ChristyBodekFit['rtie'] = 1.06*f_RTIE(ChristyBodekFit['nu'] - 0.018) # shift the peak
        axs[i,0].plot(ChristyBodekFit['nu'],(ChristyBodekFit["rlqe"]+ChristyBodekFit["rlie"]+ChristyBodekFit["rle"]+ChristyBodekFit["rlns"])*1e3,color='black',label="$R_L$(total), $R_T$(total) Christy-Bodek-2024", linestyle='solid',lw=0.8)
        axs[i,1].plot(ChristyBodekFit['nu'],(ChristyBodekFit["rtqe"]+ChristyBodekFit["rtie"]+ChristyBodekFit["rte"]+ChristyBodekFit["rtns"]+
                rt_quasi_deuteron(nus=ChristyBodekFit['nu'],q2s=ChristyBodekFit['q2'],exs=ChristyBodekFit['ex']))*1e3,
                color='black', linestyle='-',lw=0.8)
        axs[i,0].plot(ChristyBodekFit['nu'],ChristyBodekFit["rlqe"]*1e3,color='black',label="$R_L$(QE), $R_T$(QE+TE) Christy-Bodek-2024", linestyle='dotted',lw=0.8) 
        axs[i,1].plot(ChristyBodekFit['nu'],(ChristyBodekFit["rtqe"]+ChristyBodekFit["rte"])*1e3,color='black', linestyle='dotted',lw=0.8)


        for mc in MC_PLOT_LIST:
            df_rl = sheet_mc_rl.loc[(sheet_mc_rl['qv']==qvcenter) & (sheet_mc_rl['mc']==mc)]
            df_rt = sheet_mc_rt.loc[(sheet_mc_rt['qv']==qvcenter) & (sheet_mc_rt['mc']==mc)]
            if len(df_rl) > 0:
                axs[i, 0].plot(df_rl['nu'], df_rl['rl'], **LINE_STYLES[mc], label=f'$R_L$, $R_T$ {mc}')
            if len(df_rt) > 0:
                axs[i, 1].plot(df_rt['nu'], df_rt['rt'], **LINE_STYLES[mc])

        for theory in THEORY_PLOT_LIST:
            df_rl = sheet_theory_rl.loc[(sheet_theory_rl['qv']==qvcenter) & (sheet_theory_rl['theory']==theory)]
            df_rt = sheet_theory_rt.loc[(sheet_theory_rt['qv']==qvcenter) & (sheet_theory_rt['theory']==theory)]
            if len(df_rl) > 0:
                axs[i, 0].plot(df_rl['nu'], df_rl['rl'], **LINE_STYLES[theory], label=f'$R_L$, $R_T$ {theory}')
            if len(df_rt) > 0:
                axs[i, 1].plot(df_rt['nu'], df_rt['rt'], **LINE_STYLES[theory])
        
        for exp in EXP_PLOT_LIST:
            df_rl = sheet_exp_rl.loc[(sheet_exp_rl['qv']==qvcenter) & (sheet_exp_rl['experiment']==exp)]
            df_rt = sheet_exp_rt.loc[(sheet_exp_rt['qv']==qvcenter) & (sheet_exp_rt['experiment']==exp)]
            if len(df_rl) > 0:
                axs[i, 0].errorbar(df_rl['nu'], df_rl['rl'], yerr=df_rl['rlerr'], **ERRORBAR_STYLES[exp])
                axs[i, 0].scatter(df_rl['nu'], df_rl['rl'], **SCATTER_STYLES[exp], label=f'$R_L$, $R_T$ {exp}')
            if len(df_rt) > 0:
                axs[i, 1].errorbar(df_rt['nu'], df_rt['rt'], yerr=df_rt['rterr'], **ERRORBAR_STYLES[exp])
                axs[i, 1].scatter(df_rt['nu'], df_rt['rt'], **SCATTER_STYLES[exp])


        # This analysis
        df_qvbin = df_this_analysis.loc[df_this_analysis['qvcenter'] == qvcenter]
        nu = df_qvbin['nu']
        rl = df_qvbin['RL']*1e3
        rlerr = df_qvbin['RLerr']*1e3
        rt = df_qvbin['RT']*1e3
        rterr = df_qvbin['RTerr']*1e3
        axs[i, 0].scatter(nu, rl, color='red', label = '$R_L$, $R_T$ this analysis', marker='D', s=15, edgecolors='gray', linewidth=0.5)
        axs[i, 0].errorbar(nu, rl, yerr = rlerr, ecolor='red', markersize=0, capsize=0, lw=1, fmt='D', elinewidth=1, alpha=0.5) 
        axs[i, 1].scatter(nu, rt, color='red', marker='D', s=15, edgecolors='gray', linewidth=0.5)
        axs[i, 1].errorbar(nu, rt, yerr = rterr, ecolor='red', markersize=0, capsize=0, lw=1, fmt='D', elinewidth=1, alpha=0.5)



        h, l = axs[i, 0].get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
        h, l = axs[i, 1].get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

        axs[i, 0].tick_params(which='both', direction='in', top=True, right=True)
        axs[i, 0].minorticks_on()
        axs[i, 0].yaxis.set_major_formatter(scifmt)
        axs[i, 0].set_ylabel(r'$R_L$ (GeV$^{-1}$)')
        axs[i, 0].set_xlim(0, qvcenter*1.05)
        axs[i, 0].set_ylim(0, RLRT_QVPLOT_HEIGHTS[qvcenter][0])


        axs[i, 1].tick_params(which='both', direction='in', top=True, right=True)
        axs[i, 1].minorticks_on()
        axs[i, 1].yaxis.set_major_formatter(scifmt)
        axs[i, 1].set_ylabel(r'$R_T$ (GeV$^{-1}$)')
        axs[i, 1].set_xlim(0, qvcenter*1.05)
        axs[i, 1].set_ylim(0, RLRT_QVPLOT_HEIGHTS[qvcenter][1])


    axs[-1, 0].set_xlabel(r'$\nu$ (GeV)')
    axs[-1, 1].set_xlabel(r'$\nu$ (GeV)')

    unique = dict(zip(labels, handles))
    handles = list(unique.values())
    labels = list(unique.keys())

    # add a qvcenter text to subplots
    for i, qvcenter in enumerate(qvcenters):
        axs[i, 0].plot(0, 0, label='$R_L$ ($|\mathbf{q}|$ = ' + f'{qvcenter} GeV/c)', alpha=0)
        axs[i, 1].plot(0, 0, label='$R_T$ ($|\mathbf{q}|$ = ' + f'{qvcenter} GeV/c)', alpha=0)



    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06), frameon=False)

    # fig.tight_layout(rect=[0, 0.05, 1, 1])  # leave space at bottom
    fig.tight_layout()

    if figshow:
        plt.show()

    plt.close()

    return fig
    # return None


