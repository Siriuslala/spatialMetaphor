import jsonlines


def compute_circuit_overlap(task1_edges_path, task2_edges_path):
    edges_task1 = []
    edges_task2 = []
    with jsonlines.open(task1_edges_path, "r") as f:
        for line in f:
            edges_task1 = line["edges"]
    with jsonlines.open(task2_edges_path, "r") as f:
        for line in f:
            edges_task2 = line["edges"]

    # compute IoU from node level and edge level separately
    nodes_task1 = set()
    for edge in edges_task1:
        nodes_task1.add(f"up_{edge[0]}")
        nodes_task1.add(f"down_{edge[1]}")
    nodes_task2 = set()
    for edge in edges_task2:
        nodes_task2.add(f"up_{edge[0]}")
        nodes_task2.add(f"down_{edge[1]}")
    edges_task1 = [(f"up_{edge[0]}", f"down_{edge[1]}") for edge in edges_task1]
    edges_task2 = [(f"up_{edge[0]}", f"down_{edge[1]}") for edge in edges_task2]
    
    iou_node = len(nodes_task1.intersection(nodes_task2)) / len(nodes_task1.union(nodes_task2))
    iou_edge = len(set(edges_task1).intersection(set(edges_task2))) / len(set(edges_task1).union(set(edges_task2)))
    print(f"Node-level IoU: {iou_node:.4f}")
    print(f"Edge-level IoU: {iou_edge:.4f}")
   