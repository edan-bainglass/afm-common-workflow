from __future__ import annotations

import typing as t
from enum import Enum
from pathlib import Path

from aiida import orm
from aiida_quantumespresso.calculations.pp import PpCalculation
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida_quantumespresso.workflows.pw.relax import PwRelaxWorkChain
from aiida_workgraph import dynamic, namespace, shelljob, task
from ase import Atoms


class AfmCase(Enum):
    EMPIRICAL = "empirical"
    HARTREE = "hartree"
    HARTREE_RHO = "hartree_rho"


@task
def write_afm_params(params: dict) -> orm.SinglefileData:
    afm_filepath = Path.cwd() / "params.ini"
    with open(afm_filepath, "w") as config_file:
        for key, value in params.items():
            if isinstance(value, (list, tuple)):
                value = " ".join(map(str, value))
            config_file.write(f"{key} {value}\n")
    return orm.SinglefileData(file=afm_filepath.as_posix())


@task
def write_structure_file(structure: Atoms, filename: str) -> orm.SinglefileData:
    geom_filepath = Path.cwd() / filename
    structure.write(geom_filepath, format="xyz")
    return orm.SinglefileData(file=geom_filepath.as_posix())


RelaxJob = task(PwRelaxWorkChain)
ScfJob = task(PwBaseWorkChain)
PpJob = task(PpCalculation)


@task.graph
def AfmWorkflow(
    case: AfmCase,
    structure: orm.StructureData,
    afm_params: dict,
    relax: bool = False,
    dft_params: t.Annotated[
        dict[str, dict],
        namespace(
            structure=RelaxJob.inputs,
            tip=ScfJob.inputs,
        ),
    ] = None,
    pp_params: t.Annotated[
        dict[str, dict],
        namespace(
            hartree=PpJob.inputs,
            charge=PpJob.inputs,
        ),
    ] = None,
    tip: orm.StructureData = None,
) -> t.Annotated[dict, dynamic(t.Any)]:
    """AFM simulation workflow."""

    if relax:
        assert dft_params, "Missing DFT parameters"
        geom_dft_params = dft_params.get("structure", {})
        dft_job = RelaxJob(
            structure=structure,
            **geom_dft_params,
        )
        structure = dft_job.output_structure
    else:
        assert structure, "Missing structure"

    geometry_file = write_structure_file(structure, "geo.xyz").result

    assert afm_params, "Missing AFM parameters"
    afm_params_file = write_afm_params(params=afm_params).result

    ljff = shelljob(
        command="ppafm-generate-ljff",
        nodes={
            "geometry": geometry_file,
            "parameters": afm_params_file,
        },
        arguments=[
            "-i",
            "geo.xyz",
            "-f",
            "npy",
        ],
        outputs=["FFLJ.npz"],
    )

    scan_nodes = {
        "parameters": afm_params_file,
        "ljff_data": ljff.FFLJ_npz,
    }

    metadata = {
        "options": {
            "use_symlinks": True,
        }
    }

    if case != AfmCase.EMPIRICAL.name:
        if not relax:
            assert dft_params, "Missing DFT parameters"
            geom_dft_params = dft_params.get("structure", {})
            scf_params = geom_dft_params.get("base", {})
            assert scf_params, "Missing structure base SCF parameters"
            scf_params["pw.structure"] = structure
            scf_params["pw"]["parameters"]["CONTROL"]["calculation"] = "scf"
            dft_job = ScfJob(**scf_params)

        assert pp_params, "Missing post-processing parameters"
        hartree_params = pp_params.get("hartree", {})
        assert hartree_params, "Missing Hartree parameters"
        hartree_task = PpJob(
            parent_folder=dft_job.remote_folder,
            **hartree_params,
        )

        if case == AfmCase.HARTREE.name:
            elff = shelljob(
                command="ppafm-generate-elff",
                metadata=metadata,
                nodes={
                    "parameters": afm_params_file,
                    "ljff_data": ljff.FFLJ_npz,
                    "hartree_data": hartree_task.remote_folder,
                },
                filenames={
                    "hartree_data": "hartree",
                },
                arguments=[
                    "-i",
                    "hartree/aiida.fileout",
                    "-F",
                    "cube",
                    "-f",
                    "npy",
                ],
                outputs=["FFel.npz"],
            )

            scan_nodes["elff_data"] = elff.FFel_npz

        # TODO experimental feature - not fully tested - needs further attention
        elif case == AfmCase.HARTREE_RHO.name:
            charge_params = pp_params.get("charge", {})
            assert charge_params, "Missing charge density parameters"
            rho_job = PpJob(
                parent_folder=dft_job.remote_folder,
                **charge_params,
            )

            assert tip, "Missing tip structure"
            tip_scf_params = dft_params.get("tip", {})
            assert tip_scf_params, "Missing tip DFT parameters"
            tip_scf_params["pw.structure"] = tip
            tip_dft_job = ScfJob(**tip_scf_params)

            tip_rho_job = PpJob(
                parent_folder=tip_dft_job.remote_folder,
                **charge_params,
            )

            conv_rho = shelljob(
                command="ppafm-conv-rho",
                nodes={
                    "geom_density": rho_job.remote_folder,
                    "tip_density": tip_rho_job.remote_folder,
                },
                filenames={
                    "geom_density": "structure",
                    "tip_density": "tip",
                },
                arguments=[
                    "-s",
                    "structure/aiida.fileout",
                    "-t",
                    "tip/aiida.fileout",
                    "-B",
                    "1.0",
                    "-E",
                ],
            )

            charge_elff = shelljob(
                command="ppafm-generate-elff",
                nodes={
                    "conv_rho_data": conv_rho.remote_folder,
                    "hartree_data": hartree_task.remote_folder,
                    "tip_density": tip_rho_job.remote_folder,
                },
                filenames={
                    "conv_rho_data": "conv_rho",
                    "hartree_data": "hartree",
                    "tip_density": "tip",
                },
                arguments=[
                    "-i",
                    "hartree/aiida.fileout",
                    "-tip-dens",
                    "tip/aiida.fileout",
                    "--Rcode",
                    "0.7",
                    "-E",
                    "--doDensity",
                ],
                outputs=["FFel.npz"],
            )

            dftd3 = shelljob(
                command="ppafm-generate-dftd3",
                nodes={
                    "hartree_data": hartree_task.remote_folder,
                },
                filenames={
                    "hartree_data": "hartree",
                },
                arguments=[
                    "-i",
                    "hartree/aiida.fileout",
                    "--df_name",
                    "PBE",
                ],
            )

            elff = shelljob(
                command="ppafm-generate-elff",
                nodes={
                    "hartree_data": hartree_task.remote_folder,
                    "charge_elff_data": charge_elff.FFel_npz,
                },
                filenames={
                    "hartree_data": "hartree",
                },
                arguments=[
                    "-i",
                    "hartree/aiida.fileout",
                    "-f",
                    "npy",
                ],
                outputs=["FFel.npz"],
            )

        else:
            raise ValueError(f"Unsupported case: {case}")

    scan = shelljob(
        command="ppafm-relaxed-scan",
        metadata=metadata,
        nodes=scan_nodes,
        arguments=[
            "-f",
            "npy",
        ],
        outputs=["Q0.00K0.35"],
    )

    results = shelljob(
        command="ppafm-plot-results",
        metadata=metadata,
        nodes={
            "parameters": afm_params_file,
            "scan_dir": scan.Q0_00K0_35,
        },
        filenames={
            "scan_dir": "Q0.00K0.35",
        },
        arguments=[
            "--df",
            "--cbar",
            "--save_df",
            "-f",
            "npy",
        ],
        outputs=["Q0.00K0.35"],
    )

    return results
