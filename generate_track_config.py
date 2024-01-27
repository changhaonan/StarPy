"""Generate tracking config for BundleTracking"""
import os
import yaml
import open3d as o3d


if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.abspath(__file__))
    asset_dir = os.path.join(root_dir, "assets")

    bundletrack_path = os.path.join(root_dir, "external", "BundleTrack")
    default_config = os.path.join(bundletrack_path, "config_colmap.yml")
    # Read default config
    with open(default_config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # Update config
    fix_camera_id = 2
    config["data_dir"] = os.path.join(root_dir, "test_data", "exp_1", f"camera_{fix_camera_id}")
    config["mask_dir"] = os.path.join(config["data_dir"], "seg")
    config["model_dir"] = os.path.join(asset_dir, "unit_cube.obj")
    config["debug_dir"] = os.path.join(root_dir, "log")
    # Save config
    config_file = os.path.join(root_dir, "config.yml")
    with open(config_file, "w") as f:
        yaml.dump(config, f)

    # Save a unit cube obj
    unit_cube_obj = o3d.geometry.TriangleMesh.create_box(1, 1, 1)
    unit_cube_obj.compute_vertex_normals()
    unit_cube_obj_file = os.path.join(asset_dir, "unit_cube.obj")
    o3d.io.write_triangle_mesh(unit_cube_obj_file, unit_cube_obj)
