##################################################################################################
#
# network_utils.py contains utility functions for the agent model -- specifically, a neural network
# instantiated with PsyNeuLink.
#
# Author: Yotam Sagiv
#
##################################################################################################

import numpy as np
import psyneulink as pnl
import networkx as nx
import sys
import warnings
warnings.filterwarnings("error", category=UserWarning)
warnings.filterwarnings("ignore", "Pathway specified in 'pathway' arg for add_backpropagation_learning_pathway method", category=UserWarning)
warnings.filterwarnings("ignore", "Unable to copy weight matrix for", category=UserWarning)
###################### Convenience functions for testing script #################################

# read in bipartite graph, return graph object, number of possible tasks, number of
# input dimensions and number of output dimensions.
# file format Ni No (input dimension number, output dimension number)
def read_bipartite_adjlist(filename):
    g = nx.Graph()
    with open(filename) as f:
        for line in f:
            inode, onode = line.split()
            g.add_node(inode, bipartite=1)
            g.add_node(onode, bipartite=0)
            g.add_edge(inode, onode, task_id=(inode, onode))

    onodes = { n for n, d in g.nodes(data=True) if d['bipartite'] == 0 }
    inodes = set(g) - onodes

    return g, g.number_of_edges(), len(inodes), len(onodes)

############################## Network utility functions #######################################

# Given LCA parameters and environment spec, return the LCA connectivity matrix
def get_LCA_matrix(num_output_dims, num_features, self_excitation, competition):
    output_layer_size = num_output_dims * num_features

    # Set up unit block matrix
    unit_dim_mat = np.zeros((num_features, num_features)) - competition
    np.fill_diagonal(unit_dim_mat, self_excitation)

    # Build overall matrix in terms of unit block matrices on the diagonal, zeros elsewhere
    for output_dim in range(num_output_dims):
        if output_dim == 0:
            lca_mat = unit_dim_mat
        else:
            lca_mat = np.block([
                    [lca_mat, np.zeros((lca_mat.shape[0], unit_dim_mat.shape[1]))],
                    [np.zeros((unit_dim_mat.shape[0], lca_mat.shape[1])), unit_dim_mat],
            ])

    return lca_mat

# Given a bipartite graph object, parse out a list of all the tasks defined on
# that graph.
def get_all_tasks(env_bipartite_graph):
    graph_edges = env_bipartite_graph.edges()
    all_tasks = []

    # Check if the nodes are the right way around (input, output)
    for edge in graph_edges:
        if edge[1][-1] == 'i' or edge[0][-1] == 'o': # hopefully equivalent
            inode = edge[1]
            onode = edge[0]
        else:
            inode = edge[0]
            onode = edge[1]

        # Strip suffix and convert to ints
        all_tasks.append((int(inode[:-1]), int(onode[:-1])))

    return all_tasks

# Train a multitasking network using PsyNeuLink and return the trained network, with optional attached
# RecurrentTransferMechanism at the end for performance evaluation
# Params:
#     bipartite_graph: bipartite graph representing the task environment (NetworkX object)
#     num_features: number of particular features per dimension (e.g. number of colours)
#     num_hidden: number of hidden units in the network
#     epochs: number of training iterations
#     learning_rate: learning rate for SGD or (however pnl train their networks)
#     attach_LCA: if True, will attach an LCAMechanism to evaluate network performance
#     rest: LCA parameters
def get_trained_network_multLCA(bipartite_graph, num_features=3, num_hidden=200, epochs=10, learning_rate=20,
                                                                attach_LCA=True, competition=0.2, self_excitation=0.2, leak=0.4, threshold=1e-4,
                                                                exec_limit=10000, bin_execute=True, prefix=""):
    np.random.seed(12345)
    # Get all tasks from bipartite graph (edges) and strip 'i/o' suffix
    all_tasks = get_all_tasks(bipartite_graph)

    # Analyze bipartite graph for network properties
    onodes = [ n for n, d in bipartite_graph.nodes(data=True) if d['bipartite'] == 0 ]
    inodes = [ n for n, d in bipartite_graph.nodes(data=True) if d['bipartite'] == 1 ]
    input_dims = len(inodes)
    output_dims = len(onodes)
    num_tasks = len(all_tasks)

    # Start building network as PsyNeuLink object
    # Layer parameters
    nh = num_hidden
    D_i = num_features * input_dims
    D_c = num_tasks
    D_h = nh
    D_o = num_features * output_dims

    # Weight matrices (defaults provided by Dillon)
    wih = np.random.rand(D_i, D_h) * 0.02 - 0.01
    wch = np.random.rand(D_c, D_h) * 0.02 - 0.01
    wco = np.random.rand(D_c, D_o) * 0.02 - 0.01
    who = np.random.rand(D_h, D_o) * 0.02 - 0.01

    # Training params (defaults provided by Dillon)
    patience = 10
    min_delt = 0.00001
    lr = learning_rate

    # Instantiate layers and projections
    il = pnl.TransferMechanism(size=D_i, name=f'{prefix}input')
    cl = pnl.TransferMechanism(size=D_c, name=f'{prefix}control')

    hl = pnl.TransferMechanism(size=D_h,
                               name='hidden',
                               function=pnl.Logistic(bias=-2))

    ol = pnl.TransferMechanism(size=D_o,
                               name='output',
                               function=pnl.Logistic(bias=-2))

    pih = pnl.MappingProjection(matrix=wih)
    pch = pnl.MappingProjection(matrix=wch)
    pco = pnl.MappingProjection(matrix=wco)
    pho = pnl.MappingProjection(matrix=who)

    # Create training data for network
    # We train across all possible inputs, one task at a time
    input_examples, output_examples, control_examples = generate_training_data(all_tasks, num_features, input_dims, output_dims)

    # Training parameter set
    input_set = {
            'inputs': {
                    il: input_examples.tolist(),
                    cl: control_examples.tolist()
            },
            'targets': {
                    ol: output_examples.tolist()
            },
            'epochs': epochs
    }

    # Build network
    mnet = pnl.AutodiffComposition(learning_rate=learning_rate,
                                   name=f'{prefix}mnet')

    mnet.output_CIM.parameters.value._set_history_max_length(1000)
    mnet.add_node(il)
    mnet.add_node(cl)
    mnet.add_node(hl)
    mnet.add_node(ol)
    mnet.add_projection(projection=pih, sender=il, receiver=hl)
    mnet.add_projection(projection=pch, sender=cl, receiver=hl)
    mnet.add_projection(projection=pco, sender=cl, receiver=ol)
    mnet.add_projection(projection=pho, sender=hl, receiver=ol)

    # Train network
    mnet.learn(inputs=input_set,
               minibatch_size=1,
               bin_execute=bin_execute,
               patience=patience,
               min_delta=min_delt)

    for projection in mnet.projections:
        try:
            weights = projection.parameters.matrix.get(mnet)
            projection.parameters.matrix.set(weights, None)
        except AttributeError as e:
            warnings.warn(f"Unable to copy weight matrix for {projection}")

    # Apply LCA transform (values from Sebastian's code -- supposedly taken from the original LCA paper from Marius & Jay)
    if attach_LCA:
        lci = pnl.LeakyCompetingIntegrator(rate=leak,
                                           time_step_size=0.01)

        lca_matrix = get_LCA_matrix(output_dims, num_features, self_excitation, competition)

        lca = pnl.RecurrentTransferMechanism(size=D_o,
                                             matrix=lca_matrix,
                                             integrator_mode=True,
                                             integrator_function=lci,
                                             name='lca',
                                             termination_threshold=threshold,
                                             reset_stateful_function_when=pnl.AtTrialStart())

        # Wrapper composition used to pass values between mnet (AutodiffComposition) and lca (LCAMechanism)
        wrapper_composition = pnl.Composition()

        # Add mnet and lca to outer_composition
        wrapper_composition.add_linear_processing_pathway([mnet, lca])

        # Set execution limit
        lca.parameters.max_executions_before_finished.set(exec_limit, wrapper_composition)

        # # Logging/Debugging
        # lca.set_log_conditions('value', pnl.LogCondition.EXECUTION)

        return wrapper_composition

    return mnet


# Generate data for the network to train on. This means single-task training on all available
# tasks within the environment, under a uniform task distribution. As data we generate all possible
# mappings within each task. To specify a mapping, we use the rule that input feature nodes map to
# equal ordinal output feature nodes (i.e. 1st feature input maps to 1st feature output).
# Params:
#     all_tasks: list containing all tasks in the environment
#     num_features: number of particular features per dimension (e.g. number of colours)
#     num_input_dims: number of input dimensions in the environment
#     num_output_dims: number of output dimensions in the environment
#     samples_per_feature: how many stimuli will be sampled per feature to be trained within a task
#               (even though the input-output feature mapping is fixed for a given task, the values
#                of all the other input dimensions are not, so we can sample many stimuli for a given
#                feature association)
def generate_training_data(all_tasks, num_features, num_input_dims, num_output_dims, samples_per_feature=100):
    # Extract relevant parameters
    num_tasks = len(all_tasks)
    input_layer_size = num_features * num_input_dims
    output_layer_size = num_features * num_output_dims
    control_layer_size = num_tasks
    num_examples = num_features * num_tasks * samples_per_feature

    # Instantiate example matrices
    input_examples = np.zeros((num_examples, input_layer_size))
    output_examples = np.zeros((num_examples, output_layer_size))
    control_examples = np.zeros((num_examples, control_layer_size))

    # Create examples, task by task
    row_count = 0
    for task in all_tasks:
        # Load parameters
        task_input_dim, task_output_dim = task
        control_idx = task_id_to_control_idx(task, all_tasks)

        # Generate feature maps (we arbitrarily pick the redundant mapping within each dimension)
        # and also randomly sample input values for the other dimensions of the stimulus
        for _ in range(samples_per_feature):
            # Set feature of relevant task
            for i in range(num_features):
                input_examples[row_count, task_input_dim * num_features + i] = 1
                output_examples[row_count, task_output_dim * num_features + i] = 1
                control_examples[row_count, control_idx] = 1

                # Set all other stimulus dimensions randomly
                for input_dim in range(num_input_dims):
                    if input_dim == task_input_dim:
                        continue

                    input_examples[row_count, input_dim * num_features + np.random.choice(num_features)] = 1

                row_count += 1

    return input_examples, output_examples, control_examples

# Generate data for the network to test on. test_tasks is a performance set (i.e. a multitasking set of tasks to execute).
# As data we generate random features for all input dimensions. To specify a mapping, we use the rule that input
# feature nodes map to equal ordinal output feature nodes (i.e. 1st feature input maps to 1st feature output).
# Params:
#     test_tasks: list containing set of tasks to multitask
#     all_tasks: list containing all tasks in the environment
#     num_features: number of particular features per dimension (e.g. number of colours)
#     num_input_dims: number of input dimensions in the environment
#     num_output_dims: number of output dimensions in the environment
#     num_test_points: number of test points to generate
def generate_testing_data(test_tasks, all_tasks, num_features, num_input_dims, num_output_dims, num_test_points):
    # Extract relevant parameters
    num_tasks = len(all_tasks)
    input_layer_size = num_features * num_input_dims
    output_layer_size = num_features * num_output_dims
    control_layer_size = num_tasks

    # Instantiate example matrices
    input_examples = np.zeros((num_test_points, input_layer_size))
    output_examples = np.zeros((num_test_points, output_layer_size))
    control_examples = np.zeros((num_test_points, control_layer_size))

    # Create examples
    for i in range(num_test_points):
        for input_dim in range(num_input_dims):
            # Input
            feature = np.random.choice(num_features)
            input_idx = input_dim * num_features + feature
            input_examples[i, input_idx] = 1

            # Output & Control
            for output_dim in range(num_output_dims):
                # If there is not a task with this input/output pair, move on
                if (input_dim, output_dim) not in test_tasks:
                    continue

                # Output
                output_idx = output_dim * num_features + feature
                output_examples[i, output_idx] = 1

                # Control
                control_idx = task_id_to_control_idx((input_dim, output_dim), all_tasks)
                control_examples[i, control_idx] = 1

    return input_examples, output_examples, control_examples

# We define the control layer activation as just the index of task within the global all_tasks list.
def task_id_to_control_idx(task, all_tasks):
    return all_tasks.index(task)


############################## TESTING SCRIPT ##############################

# Trivial testing script
if __name__ == '__main__':
    np.set_printoptions(precision=7, threshold=sys.maxsize, suppress=True, linewidth=np.nan)
    verbose = False
    np.random.seed(12345)

    # Train and evaluate an mnet-LCA combo on single-tasking and multitasking

    num_test_points = 100
    num_features = 3
    g, num_tasks, num_input_dims, num_output_dims = read_bipartite_adjlist('./data/bipartite_graphs/7-tasks.txt')
    all_tasks = get_all_tasks(g)
    # Params
    for i in range(100, 1000, 100):
        print("TRY EPOCH", i)
        print("TRAIN PYTHON:")
        import time
        t_s = time.time()
        mnet_lca = get_trained_network_multLCA(g, learning_rate=0.3, epochs=i, attach_LCA = False, exec_limit=10000, bin_execute=False, prefix=f"{i}-python-")
        print("FINISHED IN", time.time() - t_s)

        print("TRAIN LLVM:")
        t_s = time.time()
        mnet_lca_llvm = get_trained_network_multLCA(g, learning_rate=0.3, epochs=i, attach_LCA = False, exec_limit=10000, bin_execute=True, prefix=f"{i}-llvm-")
        print("FINISHED IN", time.time() - t_s)
        for task_i in all_tasks[:2]:
            for task_j in all_tasks[:2]:
                if task_i == task_j:
                    continue
                test_tasks = [task_i, task_j]
                input_layer_size = num_features * num_input_dims
                output_layer_size = num_features * num_output_dims
                control_layer_size = len(all_tasks)
                num_tasks = len(test_tasks)

                # Instantiate test matrices
                input_test_pts, output_true_pts, control_test_pts = generate_testing_data(test_tasks,
                                                                                        all_tasks,
                                                                                        num_features,
                                                                                        num_input_dims,
                                                                                        num_output_dims,
                                                                                        num_test_points=num_test_points)

                # Run the outer composition, one point at a time (for debugging purposes)
                for j in range(num_test_points):
                    input_set = {
                                    mnet_lca.nodes[f'{i}-python-input'] : input_test_pts[j, :].tolist(),
                                    mnet_lca.nodes[f'{i}-python-control'] : control_test_pts[j, :].tolist()
                            }
                    input_set_llvm = {
                                mnet_lca_llvm.nodes[f'{i}-llvm-input'] : input_test_pts[j, :].tolist(),
                                mnet_lca_llvm.nodes[f'{i}-llvm-control'] : control_test_pts[j, :].tolist()
                        }

                    out_python = mnet_lca.run(input_set)
                    out_llvm = mnet_lca_llvm.run(input_set_llvm)
                    # Construct input dict
                    # input_set = {
                    #                 mnet_lca.nodes['mnet'].nodes['input'] : input_test_pts[i, :].tolist(),
                    #                 mnet_lca.nodes['mnet'].nodes['control'] : control_test_pts[i, :].tolist()
                    #         }
                    # input_set_llvm = {
                    #             mnet_lca_llvm.nodes['mnet-1'].nodes['input-1'] : input_test_pts[i, :].tolist(),
                    #             mnet_lca_llvm.nodes['mnet-1'].nodes['control-1'] : control_test_pts[i, :].tolist()
                    #     }

                    # out_python = mnet_lca.run( { mnet_lca.nodes['mnet'] : input_set } )
                    # out_llvm = mnet_lca_llvm.run( { mnet_lca_llvm.nodes['mnet-1'] : input_set_llvm } )
                    assert np.allclose(out_python, out_llvm)
