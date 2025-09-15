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
def write_structure_file(structure: Atoms) -> orm.SinglefileData:
    geom_filepath = Path.cwd() / "geo.xyz"
    structure.write(geom_filepath, format="xyz")
    return orm.SinglefileData(file=geom_filepath.as_posix())


RelaxationJob = task(PwRelaxWorkChain)
ScfJob = task(PwBaseWorkChain)
PpJob = task(PpCalculation)


@task.graph
def AfmWorkflow(
    case: AfmCase,
    structure: orm.StructureData,
    afm_params: dict,
    relax: bool = False,
    dft_params: t.Annotated[
        dict,
        RelaxationJob.inputs,
    ] = None,
    pp_params: t.Annotated[
        dict,
        namespace(
            hartree=PpJob.inputs,
            charge=namespace(
                structure=PpJob.inputs,
                tip=PpJob.inputs,
            ),
        ),
    ] = None,
    tip: orm.StructureData = None,
) -> t.Annotated[dict, dynamic(t.Any)]:
    """AFM simulation workflow."""

    dft_task = None
    hartree_task = None
    rho_task = None
    tip_rho_task = None

    if relax:
        assert dft_params, "Missing DFT parameters"
        dft_task = RelaxationJob(
            structure=structure,
            **dft_params,
        )
        structure = dft_task.output_structure

    assert structure, "Missing structure"
    geometry_file = write_structure_file(structure=structure).result

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

    if case != AfmCase.EMPIRICAL:
        if not relax:
            assert dft_params, "Missing DFT parameters"
            scf_params = dft_params.get("base", {})
            assert scf_params, "Missing base SCF parameters"
            scf_params["pw.structure"] = structure
            dft_task = ScfJob(**scf_params)

        assert pp_params, "Missing post-processing parameters"
        hartree_params = pp_params.get("hartree", {})
        assert hartree_params, "Missing Hartree parameters"
        hartree_task = PpJob(
            parent_folder=dft_task.remote_folder,
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

        # Experimental feature, not fully tested
        elif case == AfmCase.HARTREE_RHO:
            charge_namespace: dict = pp_params.get("charge", {})
            geom_charge_params = charge_namespace.get("structure", {})
            assert geom_charge_params, "Missing structure charge density parameters"
            rho_task = PpJob(
                structure=structure,
                parent_folder=dft_task.remote_folder,
                **geom_charge_params,
            )

            # write tip file

            tip_charge_params = charge_namespace.get("tip", {})
            assert tip, "Missing tip structure"
            assert tip_charge_params, "Missing tip charge density parameters"
            tip_rho_task = PpJob(
                structure=tip,
                parent_folder=dft_task.remote_folder,
                **tip_charge_params,
            )

            conv_rho = shelljob(
                command="ppafm-conv-rho",
                nodes={
                    "geom_density": rho_task.remote_folder,
                    "tip_density": tip_rho_task.remote_folder,
                },
                filenames={
                    "geom_density": "structure",
                    "tip_density": "tip",
                },
                arguments=[
                    "-s",
                    "structure/charge.cube",
                    "-t",
                    "tip/charge.cube",
                    "-B",
                    "1.0",
                    "-E",
                ],
                outputs=["charge.cube"],
            )

            charge_elff = shelljob(
                command="ppafm-generate-elff",
                nodes={
                    "hartree_data": hartree_task.remote_folder,
                    "conv_density": conv_rho.charge_cube,
                    "tip_density": tip_rho_task.remote_folder,
                },
                filenames={
                    "hartree_data": "hartree",
                    "tip_density": "tip",
                },
                arguments=[
                    "-i",
                    "hartree/hartree.cube",
                    "-tip-dens",
                    "tip/charge.cube",
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
                    "hartree/hartree.cube",
                    "--df_name",
                    "PBE",
                ],
                outputs=["dftd3.dat"],
            )

            elff = shelljob(
                command="ppafm-generate-elff",
                nodes={
                    "hartree_data": hartree_task.remote_folder,
                    "charge_elff_data": charge_elff.FFel_npz,
                    "dftd3_data": dftd3.dftd3_dat,
                },
                arguments=[
                    "-i",
                    "hartree/hartree.cube",
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
