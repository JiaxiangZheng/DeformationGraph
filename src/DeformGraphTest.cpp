#include <flann/flann.hpp>
#include "TriMesh.h"
#include "DeformGraph.h"
/* debug utility functions */
////////////////////////////////////////////////////////////////////////////////
static void _write_graphs(const char* filename, const std::vector<std::vector<int>>& neighs) {
	FILE* fp = fopen(filename, "w");
	if (!fp) return;
	fprintf(fp, "%d\n", neighs.size());
	for (int i=0; i<neighs.size(); ++i) {
		fprintf(fp, "%d ", neighs[i].size());
		for (int j=0; j<neighs[i].size(); ++j)
			fprintf(fp, "%d ", neighs[i][j]);
		fprintf(fp, "\n");
	}
	fclose(fp);
	return;
}
static void _save_pcd2ply(std::vector<Eigen::Vector3d> node_pos,
std::vector<Eigen::Vector3d> node_norm, const char* filename) {
    assert (node_pos.size() == node_norm.size());
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "warning : unable to open %s when saving pcd to ply file.\n", filename);
        return;
    }
    fprintf(fp, "ply\n");
    fprintf(fp, "format ascii 1.0\n");
    fprintf(fp, "element vertex %d\n", node_pos.size());
    fprintf(fp, "property float x\n");
    fprintf(fp, "property float y\n");
    fprintf(fp, "property float z\n");
    fprintf(fp, "property float nx\n");
    fprintf(fp, "property float ny\n");
    fprintf(fp, "property float nz\n");
    fprintf(fp, "property uchar red\n");
    fprintf(fp, "property uchar green\n");
    fprintf(fp, "property uchar blue\n");
    fprintf(fp, "end_header\n");
    
    for (size_t i=0; i<node_pos.size(); ++i) {
        fprintf(fp, "%f %f %f %f %f %f %d %d %d\n", 
            node_pos[i][0], node_pos[i][1], node_pos[i][2],
            node_norm[i][0], node_norm[i][1], node_norm[i][2],
            0, 255, 255);
    }
    fclose(fp);
}
static void _save_pcd2obj(std::vector<Eigen::Vector3d> node_pos,
std::vector<Eigen::Vector3d> node_norm, const char* filename) {
	TriMesh mesh;
	mesh.vert_num = node_norm.size();
	mesh.vertex_coord = node_pos;
	mesh.norm_coord = node_norm;
	mesh.saveOBJ(filename);
}
////////////////////////////////////////////////////////////////////////////////

int main() {
    // make sure these meshes are normalized, aligned rigidly and all vertex is valid (normal is valid)
	const char* src_filename = "./data/hand_000.obj";
	const char* dst_filename = "./data/hand_050.obj";
    TriMesh mesh_in(src_filename);
    TriMesh mesh_out(dst_filename);

    std::vector<int> sampling_index;
	double sampling_dist = 0.03;
    mesh_sampling(mesh_in, sampling_index, sampling_dist);

    DGraph graph;
    graph.build_graph(mesh_in, sampling_index, 2.0*sampling_dist, 8);
	_write_graphs("graph.txt", graph.node_neigh);	// for debug

	double *raw_dataset = new double[3*mesh_out.vertex_coord.size()];
    for (size_t i=0; i<mesh_out.vertex_coord.size(); ++i) {
        raw_dataset[3*i+0] = mesh_out.vertex_coord[i][0];
        raw_dataset[3*i+1] = mesh_out.vertex_coord[i][1];
        raw_dataset[3*i+2] = mesh_out.vertex_coord[i][2];
    }
    flann::Matrix<double> flann_dataset(raw_dataset, mesh_out.vertex_coord.size(), 3);
    flann::Index< flann::L2<double> > flann_index(flann_dataset, flann::KDTreeIndexParams(1));
    flann_index.buildIndex();
    flann::Matrix<double>   query_node(new double[3], 1, 3);	// num of querys * dimension
    flann::Matrix<int>      indices_node(new int[1], 1, 1);		// num of querys * knn
    flann::Matrix<double>   dists_node(new double[1], 1, 1);
    
	double max_dist_neigh = 2.0*sampling_dist;
	char save_filename[512];
	int iter = 0;
	double pre_energy = DBL_MAX;
	do {
		// 1. build nearest correspondence from graph to mesh_out
		std::vector<int> corres_indexs; std::vector<Eigen::Vector3d> corres_constraints;
		for (size_t i=0; i<graph.node_pos.size(); ++i) {
			query_node[0][0] = graph.node_pos[i][0]; query_node[0][1] = graph.node_pos[i][1]; query_node[0][2] = graph.node_pos[i][2];
			flann_index.knnSearch(query_node, indices_node, dists_node, 1, flann::SearchParams(flann::FLANN_CHECKS_UNLIMITED));
			if (dists_node[0][0] > max_dist_neigh*max_dist_neigh) continue;	//TODO: replace with a stronger condition for matching pairs
			corres_indexs.push_back(i);
			corres_constraints.push_back(mesh_out.vertex_coord[indices_node[0][0]]);
		}
		dprintf(stdout, "%d:\tfind pairs %d\n", iter, corres_indexs.size());

        // 2. use the correspondence to initialize a Deformer and call optimize_once and update graph
		double cur_energy = optimize_once(graph, corres_indexs, corres_constraints);
		dprintf(stdout, "min_energy = %lf\n", cur_energy);
		if (fabs(pre_energy-cur_energy) < 1e-5) break;
		pre_energy = cur_energy;
	} while (++iter < 50);
	sprintf(save_filename, "d_graph.ply");
	_save_pcd2ply(graph.node_pos, graph.node_norm, save_filename);

	TriMesh source_mesh(src_filename);
	graph.deform(source_mesh);
	source_mesh.savePLY("deformed_mesh.ply");
	
//	FILE* fp = fopen("trans.txt", "w");
//	for (int i=0; i<graph.node_rots.size(); ++i) {
//		fprintf(fp, "%d\n", i);
//		for (int k=0; k<3; ++k) {
//			fprintf(fp, "%lf %lf %lf %lf\n", graph.node_rots[i](k, 0), graph.node_rots[i](k, 1), graph.node_rots[i](k, 2), graph.node_trans[i][k]);
//		}
//		fprintf(fp, "\n");
//	}
//    fclose(fp);

    delete []query_node.ptr();
    delete []dists_node.ptr();
    delete []indices_node.ptr();
    delete []raw_dataset;
    return 0;
}
