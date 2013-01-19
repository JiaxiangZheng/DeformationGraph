#ifndef _DEFORM_GRAPH_HPP_
#define _DEFORM_GRAPH_HPP_
#include <Eigen/Core>
#include <vector>
using namespace std;
#include "TriMesh.h"
using namespace Eigen;

//declaration
namespace flann{
	template<typename Distance>
	class Index;

	template<class T>
	struct L2;
};
struct deform_graph {
    std::vector<Vector3d>           node_pos;
    std::vector<Vector3d>           node_norm;
    std::vector<std::vector<int>>   node_neighbours;
    std::vector<Matrix3d>           node_rot_rotation;
    std::vector<Vector3d>           node_rot_translation;
    flann::Index<flann::L2<double>>* p_kd_flann_index;
	double*							p_kd_data;
    deform_graph();
    ~deform_graph();
    void build_kdtree();
    void build_neighbours(int k = 5, double dist_threshold = .1);
	void init(const TriMesh& source_mesh);
    //using the rotations and translations to deform the graph
    void deform();
	void save_ply(const char* filename);
};

void prepare_deform_graph(const TriMesh& input_mesh, deform_graph& graph);
void prepare_deform_graph(const TriMesh& input_mesh, const std::vector<int>& corres, deform_graph& graph);

/*  
 * \param corres 
 **/
double optimize_once(const std::vector<int>& corres, 
	const std::vector<Eigen::Vector3d>& constraints, deform_graph& graph);
double optimize();

#endif/*_DEFORM_GRAPH_HPP_*/
