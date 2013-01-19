#include <fstream>
#include <iostream>
#include <algorithm>
#include <cstdlib>
using namespace std;
#pragma warning (disable:4244)
#pragma warning (disable:4018)
#pragma warning(disable: 4996)
#include "TriMesh.h"
#include "DeformGraph.hpp"

#pragma warning(push)
#pragma warning(disable: 4819)
#pragma warning(disable: 4521)
#include <pcl/keypoints/impl/uniform_sampling.hpp>
#include <pcl/filters/impl/normal_space.hpp>
#include <pcl/search/kdtree.h>
#include <pcl/common/common.h>
#pragma warning(pop)

const double THRESHOLD_DIST = 0.05;	//should be modified according to the data
const double THRESHOLD_NORM = 0.7;	
void setup_correspondence();

void sample_pcl(const TriMesh& srcmesh, std::vector<int>& out_sample_index, double sample_rate = 0.01);

int compute_correspondence(const TriMesh& srcMesh, const std::vector<int>& sampleIndex,
	const TriMesh& dstMesh, std::vector<int>& outIndex, std::vector<bool>& indexFlag) {
	//construct the kdtree
	double *dataBase = new double[3*dstMesh.vert_num];
	{
		int _index = 0;
		for (auto it = dstMesh.vertex_coord.begin();
			it != dstMesh.vertex_coord.end(); ++it) {
				dataBase[_index] = (*it)[0];
				dataBase[_index+1] = (*it)[1];
				dataBase[_index+2] = (*it)[2];
				_index += 3;
		}
	}
	flann::Matrix<double> flann_dataset(dataBase, dstMesh.vert_num, 3);
	flann::Index<flann::L2<double>> kd_flann_index(flann_dataset, flann::KDTreeIndexParams(1));
	kd_flann_index.buildIndex();

	flann::Matrix<double>   queryNode(new double[3], 1, 3);
	flann::Matrix<int>      indicesNode(new int[3*2], 3, 2);
	flann::Matrix<double>   distsNode(new double[3*2], 3, 2);
	int cnt = 0;
	for (size_t i=0; i<sampleIndex.size(); ++i) {
		queryNode[0][0] = srcMesh.vertex_coord[sampleIndex[i]][0];
		queryNode[0][1] = srcMesh.vertex_coord[sampleIndex[i]][1];
		queryNode[0][2] = srcMesh.vertex_coord[sampleIndex[i]][2];
		kd_flann_index.knnSearch(queryNode, indicesNode, distsNode, 2, flann::SearchParams(flann::FLANN_CHECKS_UNLIMITED));
		if (distsNode[0][1] > 2*THRESHOLD_DIST || dstMesh.norm_coord[indicesNode[0][1]].dot(srcMesh.norm_coord[sampleIndex[i]]) >= THRESHOLD_NORM) {
			indexFlag[i] = false;
		} else {
			outIndex[i] = indicesNode[0][1];
			indexFlag[i] = true;
			++cnt;
		}
	}
	delete []dataBase;
	delete []queryNode.ptr();
	delete []indicesNode.ptr();
	delete []distsNode.ptr();
	return cnt;
}
void save_ply_normal_with_index(const char* szFileName, TriMesh& mesh, 
	std::vector<int>& indices);
void repair_tri_mesh(TriMesh& mesh) {
	TriMesh temp_mesh;
	temp_mesh.vertex_coord.reserve(mesh.vert_num);
	temp_mesh.norm_coord.reserve(mesh.vert_num);
	for (int i=0; i<mesh.vert_num; ++i)	{
		if (mesh.norm_coord[i].squaredNorm() < 1e-10) continue;
		temp_mesh.vertex_coord.push_back(mesh.vertex_coord[i]);
		temp_mesh.norm_coord.push_back(mesh.norm_coord[i]);
	}
	temp_mesh.vert_num = temp_mesh.vertex_coord.size();
	mesh = temp_mesh;
}
int main() {
	TriMesh inputMesh("./hand_00000.obj");
	inputMesh.polyIndex.clear();
	TriMesh targetMesh("./hand_00050.obj");
	targetMesh.polyIndex.clear();
	printf("%d-%d\n", inputMesh.vert_num, targetMesh.vert_num);
	repair_tri_mesh(inputMesh);
	repair_tri_mesh(targetMesh);
	printf("%d-%d\n", inputMesh.vert_num, targetMesh.vert_num);

	std::vector<int> simplifiedIndex;
	sample_pcl(inputMesh, simplifiedIndex); //get result
	deform_graph graph;
	prepare_deform_graph(inputMesh, simplifiedIndex, graph);
	printf("graph node number : %d\n", graph.node_pos.size());
	save_ply_normal_with_index("sim.ply", inputMesh, simplifiedIndex);
	
	std::vector<bool> flags(simplifiedIndex.size(), false);
	std::vector<int> corres_index(simplifiedIndex.size(), -1);
	int corres_cnt = compute_correspondence(inputMesh, simplifiedIndex, targetMesh, corres_index, flags);
	printf("corres number : %d\n", corres_cnt);

	std::vector<int> corres_dgraph; corres_dgraph.reserve(corres_cnt);
	std::vector<Eigen::Vector3d> constraints_coord;  constraints_coord.reserve(corres_cnt);
	for (int i=0, cnt = 0; i<simplifiedIndex.size(); ++i) {
		if (flags[i]) {
			corres_dgraph.push_back(i);
			constraints_coord.push_back(targetMesh.vertex_coord[corres_index[i]]);
		}
	}
	optimize_once(corres_dgraph, constraints_coord, graph);
	return 0;
}

static pcl::PointCloud<pcl::PointNormal>::Ptr
trimesh2pcl_data(const TriMesh& trimesh) {	
//convert the trimesh to the point cloud data with coordinates and normals.
    pcl::PointCloud<pcl::PointNormal>::Ptr out(new pcl::PointCloud<pcl::PointNormal>);
	out->points.reserve(trimesh.vert_num);
    int vi = 0;
    for (int i=0; i<trimesh.vert_num; ++i) {
        pcl::PointNormal temp_v;
        temp_v.x = trimesh.vertex_coord[i][0];
        temp_v.y = trimesh.vertex_coord[i][1];
        temp_v.z = trimesh.vertex_coord[i][2];
        temp_v.normal_x = trimesh.norm_coord[i][0];
        temp_v.normal_y = trimesh.norm_coord[i][1];
        temp_v.normal_z = trimesh.norm_coord[i][2];
        out->points.push_back(temp_v);
        ++vi;
    }
    out->width = vi; out->height = 1; out->is_dense = false;
    return out;
}

void sample_pcl(const TriMesh& srcmesh, std::vector<int>& out_sample_index, double sample_rate) {
	pcl::PointCloud<pcl::PointNormal>::Ptr src_pcd = trimesh2pcl_data(srcmesh);
	std::vector<int> normal_sampling_indices;
	std::vector<int> uniform_sampling_indices;

	const int NORMAL_SAMPLE_BIN_NUMBER = 50; 
	const int NORMAL_SAMPLE_NUMBER = 500;
	const double UNIFORM_SAMPLE_RADIUS = THRESHOLD_DIST/10;
	{
		pcl::NormalSpaceSampling<pcl::PointNormal, pcl::Normal> normal_sampler;
		pcl::PointCloud<pcl::Normal>::Ptr tri_normal(new pcl::PointCloud<pcl::Normal>);
		{
			tri_normal->resize(src_pcd->points.size());
			for (int i=0; i<src_pcd->points.size(); ++i) {
				tri_normal->points[i].normal_x = src_pcd->points[i].normal_x;
				tri_normal->points[i].normal_y = src_pcd->points[i].normal_y;
				tri_normal->points[i].normal_z = src_pcd->points[i].normal_z;
			}
		}
		normal_sampler.setSeed(time(NULL));
		normal_sampler.setInputCloud(src_pcd);
		normal_sampler.setNormals(tri_normal);
		normal_sampler.setBins(NORMAL_SAMPLE_BIN_NUMBER, NORMAL_SAMPLE_BIN_NUMBER, NORMAL_SAMPLE_BIN_NUMBER);
		normal_sampler.setSample(NORMAL_SAMPLE_NUMBER);
		normal_sampler.filter(normal_sampling_indices);
	}
	{
		typedef pcl::PointXYZ PointType;
		pcl::UniformSampling<PointType> uniform_sampler;
		pcl::PointCloud<PointType>::Ptr tri_vertex(new pcl::PointCloud<PointType>);
		{
			tri_vertex->resize(src_pcd->points.size());
			for (int i=0; i<src_pcd->points.size(); ++i) {
				tri_vertex->points[i].x = src_pcd->points[i].x;
				tri_vertex->points[i].y = src_pcd->points[i].y;
				tri_vertex->points[i].z = src_pcd->points[i].z;
			}
		}
		pcl::search::KdTree<PointType>::Ptr tree(new pcl::search::KdTree<PointType>());
		uniform_sampler.setInputCloud(tri_vertex);
		uniform_sampler.setSearchMethod(tree);
		uniform_sampler.setRadiusSearch(UNIFORM_SAMPLE_RADIUS);
		{
			pcl::PointCloud<int> keypoints_src_idx;
			uniform_sampler.compute(keypoints_src_idx);
			uniform_sampling_indices.clear();
			uniform_sampling_indices.resize(keypoints_src_idx.size());
			std::copy(keypoints_src_idx.begin(), keypoints_src_idx.end(), uniform_sampling_indices.begin());
		}
	}

	//merge the sampling result to output
    out_sample_index.clear();
	out_sample_index.resize(uniform_sampling_indices.size() + normal_sampling_indices.size());
	std::sort(uniform_sampling_indices.begin(), uniform_sampling_indices.end());
	std::sort(normal_sampling_indices.begin(), normal_sampling_indices.end());

    std::merge(uniform_sampling_indices.begin(), uniform_sampling_indices.end(),
        normal_sampling_indices.begin(), normal_sampling_indices.end(), 
		out_sample_index.begin());
	std::unique(out_sample_index.begin(), out_sample_index.end());
	printf("****uniform sampling count : %d\n****normal sampling count : %d\n", uniform_sampling_indices.size(),
		normal_sampling_indices.size());
	printf("****sampling count : %d\n", out_sample_index.size());
    return;
}

void save_ply_normal_with_index(const char* szFileName, TriMesh& mesh, 
	std::vector<int>& indices)
{
	FILE* fpOut = fopen(szFileName, "w");
	fprintf(fpOut, "ply\n");
	fprintf(fpOut, "format ascii 1.0\n");
	fprintf(fpOut, "element vertex %d\n", indices.size());
	fprintf(fpOut, "property float x\n");
	fprintf(fpOut, "property float y\n");
	fprintf(fpOut, "property float z\n");
	fprintf(fpOut, "property float nx\n");
	fprintf(fpOut, "property float ny\n");
	fprintf(fpOut, "property float nz\n");
	fprintf(fpOut, "property uchar red\n");
	fprintf(fpOut, "property uchar green\n");
	fprintf(fpOut, "property uchar blue\n");
	fprintf(fpOut, "end_header\n");

	for(int i=0; i<indices.size(); ++i)
	{
		fprintf(fpOut, "%f %f %f %f %f %f %d %d %d\n", 
			mesh.vertex_coord[indices[i]][0], mesh.vertex_coord[indices[i]][1], mesh.vertex_coord[indices[i]][2], 
			mesh.norm_coord[indices[i]][0], mesh.norm_coord[indices[i]][1], mesh.norm_coord[indices[i]][2],
			255, 0, 0);
		fflush(fpOut);
	}
	fclose(fpOut);
}
