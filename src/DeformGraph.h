#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include "TriMesh.h"

#define dprintf fprintf

struct DGraph {
    const TriMesh* p_mesh;                            // for data backup 
    std::vector<int> node_index;

    std::vector<Eigen::Vector3d> node_pos;
    std::vector<Eigen::Vector3d> node_norm;

    std::vector<std::vector<int>> node_neigh;
    std::vector<Eigen::Matrix3d> node_rots;
    std::vector<Eigen::Vector3d> node_trans;

    double max_dist_neigh;      // used for building neighbours
    int    k_neigh;
    DGraph() {
        p_mesh = NULL;
        max_dist_neigh = 0.0;
        k_neigh = 0;
    }
    void build_graph(const TriMesh& mesh, std::vector<int>& sample_index, 
        double _max_dist_neigh = 0.1, int k_nn = 10);
    void build_neigh();
	void update_graph(const Eigen::VectorXd& X);
	void deform(TriMesh& d_mesh);
};

struct Deformer {
    DGraph* p_graph;
    std::vector<double> weights;    //rot, smooth, regular
    std::vector<int> constraints_index;
    std::vector<Eigen::Vector3d> node_constraints;

    void init(DGraph& graph, const std::vector<int>& corr_indexs, 
        const std::vector<Eigen::Vector3d>& cons);
    void set_weights(double* _weights, int n) {
        weights.resize(n); 
		for (int i=0; i<n; ++i) weights[i] = _weights[i];
    }
    void build_jacobi(const Eigen::VectorXd& x, Eigen::MatrixXd& fJac);
    void build_jacobi(const Eigen::VectorXd& x, Eigen::SparseMatrix<double>& fJac);
    void build_values(const Eigen::VectorXd& x, Eigen::VectorXd& fVec);

    int rows, cols;
};

void mesh_sampling(const TriMesh& mesh, std::vector<int>& sample_index, double max_dist);
double minimize_guass_newton(Deformer& deformer, Eigen::VectorXd& X);
double minimize_levenberg_marquardt(Deformer& deformer, Eigen::VectorXd& X);
double optimize_once(DGraph& graph, std::vector<int>& constraints_index,
	const std::vector<Eigen::Vector3d>& cons); //constraints_index is in respect to graph nodes' index

