import sys
import heapq
from collections import deque

class Graph:
    def __init__(self):
        self.nodes = {}  # Node ID -> (x, y) coordinates
        self.edges = {}  # (from_node, to_node) -> cost
        self.origin = None
        self.destinations = []

    def add_node(self, node_id, x, y):
        self.nodes[node_id] = (x, y)

    def add_edge(self, from_node, to_node, cost):
        self.edges[(from_node, to_node)] = cost

    def set_origin(self, origin):
        self.origin = origin

    def set_destinations(self, destinations):
        self.destinations = destinations

    def heuristic(self, node, goal):
        """Heuristic function (Euclidean distance)."""
        x1, y1 = self.nodes[node]
        x2, y2 = self.nodes[goal]
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    def get_neighbors(self, node):
        """Returns a list of neighbors and their costs."""
        return [(to, cost) for (frm, to), cost in self.edges.items() if frm == node]

def bfs(graph):
    """Breadth-First Search (BFS)"""
    queue = deque([(graph.origin, [graph.origin])])
    visited = set()

    while queue:
        node, path = queue.popleft()
        if node in visited:
            continue
        visited.add(node)

        if node in graph.destinations:
            return path, len(visited)

        for neighbor, _ in graph.get_neighbors(node):
            if neighbor not in visited:
                queue.append((neighbor, path + [neighbor]))

    return None, len(visited)

def dfs(graph):
    """Depth-First Search (DFS)"""
    stack = [(graph.origin, [graph.origin])]
    visited = set()

    while stack:
        node, path = stack.pop()
        if node in visited:
            continue
        visited.add(node)

        if node in graph.destinations:
            return path, len(visited)

        for neighbor, _ in graph.get_neighbors(node):
            if neighbor not in visited:
                stack.append((neighbor, path + [neighbor]))

    return None, len(visited)

def greedy_best_first_search(graph):
    """Greedy Best-First Search (GBFS)"""
    queue = [(graph.heuristic(graph.origin, graph.destinations[0]), graph.origin, [graph.origin])]
    visited = set()

    while queue:
        _, node, path = heapq.heappop(queue)
        if node in visited:
            continue
        visited.add(node)

        if node in graph.destinations:
            return path, len(visited)

        for neighbor, _ in graph.get_neighbors(node):
            if neighbor not in visited:
                heapq.heappush(queue, (graph.heuristic(neighbor, graph.destinations[0]), neighbor, path + [neighbor]))

    return None, len(visited)

def a_star_search(graph):
    """A* Search"""
    queue = [(0, graph.origin, [graph.origin], 0)]  # (f, node, path, g)
    visited = {}

    while queue:
        _, node, path, g = heapq.heappop(queue)
        if node in visited and visited[node] <= g:
            continue
        visited[node] = g

        if node in graph.destinations:
            return path, len(visited)

        for neighbor, cost in graph.get_neighbors(node):
            new_g = g + cost
            f = new_g + graph.heuristic(neighbor, graph.destinations[0])
            heapq.heappush(queue, (f, neighbor, path + [neighbor], new_g))

    return None, len(visited)

def cus1_search(graph):
    """Custom Search 1 - Placeholder for an uninformed search method"""
    return bfs(graph)  # Example: BFS (Replace with your own algorithm)

def cus2_search(graph):
    """Custom Search 2 - Placeholder for an informed search method"""
    return a_star_search(graph)  # Example: A* (Replace with your own algorithm)

def load_graph_from_file(filename):
    """Reads the graph from a text file."""
    graph = Graph()
    with open(filename, 'r') as file:
        section = None
        for line in file:
            line = line.strip()
            if not line:
                continue
            if line.startswith("Nodes:"):
                section = "nodes"
                continue
            elif line.startswith("Edges:"):
                section = "edges"
                continue
            elif line.startswith("Origin:"):
                graph.set_origin(int(line.split(":")[1].strip()))
                continue
            elif line.startswith("Destinations:"):
                graph.set_destinations(list(map(int, line.split(":")[1].split(";"))))
                continue

            if section == "nodes":
                node_id, coords = line.split(":")
                x, y = map(int, coords.strip("()").split(","))
                graph.add_node(int(node_id), x, y)
            elif section == "edges":
                edge_info, cost = line.split(":")
                from_node, to_node = map(int, edge_info.strip("()").split(","))
                graph.add_edge(from_node, to_node, int(cost))

    return graph

def main():
    if len(sys.argv) != 3:
        print("Usage: python search.py <filename> <method>")
        return

    filename, method = sys.argv[1], sys.argv[2].lower()
    graph = load_graph_from_file(filename)

    search_methods = {
        "bfs": bfs,
        "dfs": dfs,
        "gbfs": greedy_best_first_search,
        "as": a_star_search,
        "cus1": cus1_search,
        "cus2": cus2_search
    }

    if method not in search_methods:
        print("Invalid method! Choose from bfs, dfs, gbfs, as, cus1, cus2")
        return

    path, explored = search_methods[method](graph)
    if path:
        print(f"{filename} {method}")
        print(f"{path[-1]} {explored}")
        print(" -> ".join(map(str, path)))
    else:
        print("No path found.")

if __name__ == "__main__":
    main()