import tempfile
import unittest

import numpy as np
import MDAnalysis as mda
from MDAnalysis.coordinates.memory import MemoryReader

from src.mdchat.context import AnalysisContext
from src.mdchat.skills.metrics import (
    ComputePCASkill,
    ComputeRMSDSkill,
    ComputeRMSFSkill,
)
from src.mdchat.skills.trajectory_observer_pass import RunTrajectoryObserverPassSkill


def _build_synthetic_universe(n_frames: int = 5) -> mda.Universe:
    n_atoms = 3
    atom_resindex = np.array([0, 0, 1], dtype=int)
    u = mda.Universe.empty(
        n_atoms,
        n_residues=2,
        atom_resindex=atom_resindex,
        trajectory=True,
    )
    u.add_TopologyAttr("name", ["A1", "A2", "B1"])
    u.add_TopologyAttr("resname", ["RES", "LIG"])
    u.add_TopologyAttr("resid", [1, 2])

    coords = np.zeros((n_frames, n_atoms, 3), dtype=np.float64)
    for i in range(n_frames):
        shift = float(i) * 0.1
        coords[i, 0, :] = [0.0 + shift, 0.0, 0.0]
        coords[i, 1, :] = [1.0 + shift, 0.0, 0.0]
        coords[i, 2, :] = [2.0 + shift, 0.2 * shift, 0.0]

    u.load_new(coords, format=MemoryReader)
    return u


class ObserverPassIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.u = _build_synthetic_universe(n_frames=6)
        self.tmpdir = tempfile.mkdtemp(prefix="mdchat_test_")
        self.context = AnalysisContext(output_dir=self.tmpdir)
        self.context.universe = self.u
        self.context.main_selection = "all"

    def test_observer_pass_collective_metrics_single_iterate(self) -> None:
        skill = RunTrajectoryObserverPassSkill()
        result = skill.execute(
            self.context,
            include_rmsd=True,
            rmsd_selection="all",
            include_rg=True,
            rg_selection="all",
            include_contacts=True,
            contact_selection_a="name A1",
            contact_selection_b="name B1",
            include_rmsf=True,
            rmsf_selection="all",
            include_pca=True,
            pca_selection="all",
            pca_n_components=2,
            save_csv=False,
            n_jobs=4,  # should be coerced to 1 for accumulation safety
        )

        self.assertTrue(result.success, msg=result.error or result.summary)
        self.assertIn("observer_pass_results", result.data)

        rmsd = self.context.get("rmsd_array")
        rg = self.context.get("rg_array")
        contacts = self.context.get("contact_distances")
        rmsf = self.context.get("rmsf_array")
        pca_scores = self.context.get("pca_scores")
        pca_var = self.context.get("pca_variance_explained")
        atom_info = self.context.get("rmsf_atom_info")
        pass_results = self.context.get("observer_pass_results")

        self.assertEqual(rmsd.shape, (6,))
        self.assertEqual(rg.shape, (6,))
        self.assertEqual(contacts.shape, (6,))
        self.assertEqual(rmsf.shape, (3,))
        self.assertEqual(pca_scores.shape, (6, 2))
        self.assertEqual(pca_var.shape, (2,))
        self.assertEqual(len(atom_info), 3)
        self.assertEqual(pass_results["n_frames_iterated"], 6)
        self.assertEqual(pass_results["n_jobs"], 1)

    def test_individual_skills_delegate_successfully(self) -> None:
        rmsd_res = ComputeRMSDSkill().execute(
            self.context,
            selection="all",
            ref_frame=0,
            save_csv=False,
        )
        rmsf_res = ComputeRMSFSkill().execute(
            self.context,
            selection="all",
            save_csv=False,
        )
        pca_res = ComputePCASkill().execute(
            self.context,
            selection="all",
            n_components=2,
            save_csv=False,
        )

        self.assertTrue(rmsd_res.success, msg=rmsd_res.error or rmsd_res.summary)
        self.assertTrue(rmsf_res.success, msg=rmsf_res.error or rmsf_res.summary)
        self.assertTrue(pca_res.success, msg=pca_res.error or pca_res.summary)

        self.assertEqual(self.context.get("rmsd_array").shape, (6,))
        self.assertEqual(self.context.get("rmsf_array").shape, (3,))
        self.assertEqual(self.context.get("pca_scores").shape, (6, 2))


if __name__ == "__main__":
    unittest.main()

