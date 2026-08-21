from __future__ import annotations

import numpy as np
import trimesh

from pose_point_depth_mv.bunny_review.finalize import (
    apply_display_transform,
    display_transform_from_mesh,
    mesh_stats,
)
from pose_point_depth_mv.render_direct_slat_fourway import (
    LATENT_DECODER_TO_REFERENCE,
    affine_audit,
    direct_method_identity,
    identity_alignment,
    parse_method_ids,
    rigid_icp,
    review_mode_ids,
)
from pose_point_depth_mv.compare_pixal3d_singleview_smoke import similarity_icp


def test_reference_display_transform_is_shared_not_refit() -> None:
    reference = trimesh.creation.box(extents=(2.0, 1.0, 0.5))
    reference.apply_translation((3.0, -2.0, 1.0))
    transform = display_transform_from_mesh(reference, 0.9, owner="reference")
    normalized_reference, reference_audit = apply_display_transform(
        reference,
        transform,
    )
    assert np.allclose(normalized_reference.bounds.mean(axis=0), 0.0)
    assert np.isclose(np.max(normalized_reference.extents), 0.9)
    assert reference_audit["owner"] == "reference"

    prediction = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    prediction.apply_translation((6.0, -2.0, 1.0))
    shared_prediction, prediction_audit = apply_display_transform(
        prediction,
        transform,
    )
    # A shared Reference transform must preserve this method's translation
    # error instead of independently recentering it.
    assert not np.allclose(shared_prediction.bounds.mean(axis=0), 0.0)
    assert prediction_audit["matrix"] == transform["matrix"]


def test_affine_audit_rejects_reflection() -> None:
    reflected = np.eye(4, dtype=np.float64)
    reflected[0, 0] = -1.0
    try:
        affine_audit(reflected)
    except ValueError as error:
        assert "reflected" in str(error)
    else:
        raise AssertionError("reflection must be rejected")


def test_identity_alignment_is_not_gt_assisted() -> None:
    audit = identity_alignment(policy="unit test")
    assert audit["gt_assisted"] is False
    assert audit["proper"] is True
    assert audit["anisotropy_ratio"] == 1.0
    assert np.allclose(np.asarray(audit["matrix"]), np.eye(4))


def test_decoder_to_reference_matches_vendored_transform_pose() -> None:
    vertices = np.array(
        [
            [1.0, 2.0, 3.0],
            [-4.0, 5.0, -6.0],
        ],
        dtype=np.float64,
    )
    vendored_row_vector_matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    expected = vertices @ vendored_row_vector_matrix
    homogeneous = np.concatenate(
        [vertices, np.ones((len(vertices), 1), dtype=np.float64)],
        axis=1,
    )
    actual = (
        LATENT_DECODER_TO_REFERENCE @ homogeneous.T
    ).T[:, :3]
    assert np.allclose(actual, expected)
    assert np.allclose(actual[:, 0], vertices[:, 0])
    assert np.allclose(actual[:, 1], vertices[:, 2])
    assert np.allclose(actual[:, 2], -vertices[:, 1])


def test_shape_alignment_is_proper_and_isotropic() -> None:
    target = trimesh.creation.icosphere(subdivisions=1, radius=0.4)
    target.vertices[:, 0] *= 1.7
    source = target.copy()
    transform = trimesh.transformations.rotation_matrix(
        np.pi / 2.0,
        (0.0, 0.0, 1.0),
    )
    transform[:3, :3] *= 1.4
    transform[:3, 3] = (0.3, -0.2, 0.1)
    source.apply_transform(transform)
    _, alignment = similarity_icp(
        source,
        target,
        seed=7,
        candidate_samples=100,
        final_samples=200,
        candidate_iterations=3,
        final_iterations=5,
    )
    assert alignment["determinant"] > 0.0
    assert alignment["anisotropy_ratio"] < 1.0 + 1.0e-6
    assert np.isfinite(alignment["cost"])


def test_rigid_alignment_changes_pose_but_not_scale() -> None:
    target = trimesh.creation.box(extents=(1.0, 0.6, 0.3))
    source = target.copy()
    transform = trimesh.transformations.rotation_matrix(
        np.pi / 2.0,
        (0.0, 0.0, 1.0),
    )
    transform[:3, 3] = (0.3, -0.2, 0.1)
    source.apply_transform(transform)
    aligned, alignment = rigid_icp(
        source,
        target,
        seed=11,
        candidate_samples=200,
        final_samples=400,
        candidate_iterations=4,
        final_iterations=8,
    )
    assert alignment["determinant"] > 0.0
    assert alignment["scale_applied"] is False
    assert np.allclose(alignment["singular_values"], np.ones(3), atol=1.0e-6)
    assert np.isclose(aligned.volume, source.volume, rtol=1.0e-6, atol=1.0e-9)


def test_pixal_pose_policy_selects_unambiguous_review_modes() -> None:
    assert review_mode_ids("metadata") == ["canonical_pose", "shape_aligned"]
    assert review_mode_ids("reference_rigid_icp") == ["pixal_pose_aligned"]


def test_direct_method_identity_uses_report_checkpoint_step() -> None:
    identity = direct_method_identity(
        {
            "format": "pose_point_depth_mv.direct_slat_mesh_exploratory.v2",
            "checkpoint_step": 200,
        },
        view_count=4,
    )
    assert identity["method_id"] == "direct_ss900_slat_step000200"
    assert identity["label"] == "Direct SS900 + SLAT step200 (4 views)"
    assert identity["checkpoint_step"] == 200


def test_partial_method_selection_uses_fixed_canonical_order() -> None:
    assert parse_method_ids("direct,reference,reconviagen_stock") == [
        "reference",
        "reconviagen_stock",
        "direct",
    ]
    assert parse_method_ids(
        "reference,pixal3d_native,reconviagen_stock,direct"
    ) == [
        "reference",
        "pixal3d_native",
        "reconviagen_stock",
        "direct",
    ]


def test_partial_method_selection_rejects_duplicates_and_unknowns() -> None:
    for value in ("reference,reference", "reference,legacy_pixal", ""):
        try:
            parse_method_ids(value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid method selector must be rejected: {value!r}")


def test_mesh_stats_counts_components_without_materializing_split_meshes() -> None:
    class DummyMesh:
        vertices = np.zeros((4, 3), dtype=np.float64)
        faces = np.zeros((2, 3), dtype=np.int64)
        body_count = 7
        is_watertight = False
        is_winding_consistent = True
        euler_number = 0
        bounds = np.zeros((2, 3), dtype=np.float64)
        extents = np.zeros(3, dtype=np.float64)

        def split(self, *args, **kwargs):
            raise AssertionError("mesh_stats must not materialize connected components")

    assert mesh_stats(DummyMesh())["components"] == 7


if __name__ == "__main__":
    test_reference_display_transform_is_shared_not_refit()
    test_affine_audit_rejects_reflection()
    test_identity_alignment_is_not_gt_assisted()
    test_decoder_to_reference_matches_vendored_transform_pose()
    test_shape_alignment_is_proper_and_isotropic()
    test_rigid_alignment_changes_pose_but_not_scale()
    test_pixal_pose_policy_selects_unambiguous_review_modes()
    test_direct_method_identity_uses_report_checkpoint_step()
    test_partial_method_selection_uses_fixed_canonical_order()
    test_partial_method_selection_rejects_duplicates_and_unknowns()
    test_mesh_stats_counts_components_without_materializing_split_meshes()
    print("render_direct_slat_fourway tests passed")
