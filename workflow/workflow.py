from enum import Enum
from pathlib import Path

from aiida import orm
from aiida_quantumespresso.calculations.pp import PpCalculation
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida_quantumespresso.workflows.pw.relax import PwRelaxWorkChain
from aiida_workgraph import shelljob, task


class AfmCase(Enum):
    EMPIRICAL = "empirical"
    HARTREE = "hartree"
    HARTREE_RHO = "hartree_rho"


def write_params(params: dict, filepath: Path):
    with open(filepath, "w") as config_file:
        for key, value in params.items():
            config_file.write(f"{key} {value}\n")


@task.graph
def AfmWorkflow(
    case: AfmCase,
    structure: orm.StructureData,
    afm_params: dict,
    relax: bool = False,
    pw_params: dict = None,
    pp_hartree_params: dict = None,
    pp_rho_params: dict = None,
    tip: orm.StructureData = None,
):
    pw_task = None
    hartree_task = None
    rho_task = None
    tip_rho_task = None

    if relax:
        assert pw_params
        pw_task = task(PwRelaxWorkChain)(
            structure=structure,
            **pw_params,
        )
        structure = pw_task.output_structure

    if case != AfmCase.EMPIRICAL:
        if not relax:
            pw_task = task(PwBaseWorkChain)(
                structure=structure,
                **pw_params,
            )

        hartree_task = task(PpCalculation)(
            structure=structure,
            parent_folder=pw_task.remote_folder,
            **pp_hartree_params,
        )

        if case == AfmCase.HARTREE_RHO:
            rho_task = task(PpCalculation)(
                structure=structure,
                parent_folder=pw_task.remote_folder,
                **pp_rho_params,
            )
            tip_rho_task = task(PpCalculation)(
                structure=tip,
                parent_folder=pw_task.remote_folder,
                **pp_rho_params,
            )

    geom_filepath = Path.cwd() / "geo.xyz"
    structure.get_ase().write(geom_filepath, format="xyz")
    geometry_file = orm.SinglefileData(file=geom_filepath.as_posix())

    afm_filepath = Path.cwd() / "params.ini"
    write_params(afm_params, afm_filepath)
    afm_params_file = orm.SinglefileData(file=afm_filepath.as_posix())

    if case == AfmCase.EMPIRICAL:
        ljff = shelljob(
            command="ppafm-generate-ljff",
            arguments=["-i geo.xyz", "-f npy"],
            nodes={
                "geometry": geometry_file,
                "parameters": afm_params_file,
            },
        )
        scan = shelljob(
            command="ppafm-relaxed-scan",
            arguments=["-f npy"],
            nodes={
                "parameters": afm_params_file,
                "ljff_data": ljff.remote_folder,
            },
        )
        results = shelljob(
            command="ppafm-plot-results",
            arguments=["--df", "--cbar", "--save_df", "-f npy"],
            nodes={
                "parameters": afm_params_file,
                "scan_data": scan.remote_folder,
            },
        )

    elif case == AfmCase.HARTREE:
        ljff = shelljob(
            command="ppafm-generate-ljff",
            nodes={
                "input": geometry_file,
            },
            arguments=["-i geo.xyz", "-f npy"],
        )
        elff = shelljob(
            command="ppafm-generate-elff",
            nodes={
                "hartree_data": hartree_task.remote_folder,
            },
            arguments=["-i hartree.cube", "-f npy"],
        )
        scan = shelljob(
            command="ppafm-relaxed-scan",
            nodes={
                "ljff_data": ljff.remote_folder,
                "elff_data": elff.remote_folder,
            },
            arguments=["-f npy"],
        )
        results = shelljob(
            command="ppafm-plot-results",
            nodes={
                "scan_data": scan.remote_folder,
            },
            arguments=["--df", "--cbar", "--save_df", "-f npy"],
        )
    elif case == AfmCase.HARTREE_RHO:
        conv_rho = shelljob(
            command="ppafm-conv-rho",
            nodes={
                "geom_density": rho_task.remote_folder,
                "tip_density": tip_rho_task.remote_folder,
            },
            arguments=[
                "-s charge.xsf",
                "-t density_CO.xsf",
                "-B 1.0",
                "-E",
            ],
        )
        charge_elff = shelljob(
            command="ppafm-generate-elff",
            nodes={
                "hartree_data": hartree_task.remote_folder,
                "tip_density": tip_rho_task.remote_folder,
            },
            arguments=[
                "-i LOCPOT.xsf",
                "-tip-dens density_CO.xsf",
                "--Rcode 0.7",
                "-E",
                "--doDensity",
            ],
        )
        dftd3 = shelljob(
            command="ppafm-generate-dftd3",
            nodes={
                "hartree_data": hartree_task.remote_folder,
            },
            arguments=["-i LOCPOT.xsf", "--df_name PBE"],
        )
        ljff = shelljob(
            command="ppafm-generate-ljff",
            nodes={
                "input": geometry_file,
            },
            arguments=["-i geo.xyz", "-f npy"],
        )
        elff = shelljob(
            command="ppafm-generate-elff",
            nodes={
                "hartree_data": hartree_task.remote_folder,
            },
            arguments=["-i hartree.cube", "-f npy"],
        )
        scan = shelljob(
            command="ppafm-relaxed-scan",
            nodes={
                "ljff_data": ljff.remote_folder,
                "elff_data": elff.remote_folder,
            },
            arguments=["-f npy"],
        )
        results = shelljob(
            command="ppafm-plot-results",
            nodes={
                "scan_data": scan.remote_folder,
            },
            arguments=["--df", "--cbar", "--save_df", "-f npy"],
        )
