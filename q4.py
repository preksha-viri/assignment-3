"""
graph_realworld.py

Contains implementations for:
- Problem 1: Social Network Friend Suggestion (BFS)
- Problem 2: Route Finding (Bellman-Ford)
- Problem 3: Emergency Response (Dijkstra)
- Problem 4: Network Cable Installation (Prim's & Kruskal's)

Also includes simple profiling (time + memory) and plotting utilities.

Requirements:
    pip install memory_profiler matplotlib

Run:
    python graph_realworld.py
"""

import heapq
import random
import time
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Set
try:
    from memory_profiler import memory_usage
except Exception:
    memory_usage = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def suggest_friends_bfs(graph: Dict[str, List[str]], user: str) -> List[str]:
    """
    Suggest friends for `user` by finding friends-of-friends not already friends.
    Uses BFS to depth 2.
    """
    if user not in graph:
        return []
    visited: Set[str] = set([user])
    q = deque([user])
    level = 0
    suggestions: Set[str] = set()

    while q and level < 2:
        for _ in range(len(q)):
            cur = q.popleft()
            for neigh in graph.get(cur, []):
                if neigh not in visited:
                    q.append(neigh)
                    visited.add(neigh)
                    # level == 1 means neigh is friend-of-friend relative to user
                    if level == 1 and neigh not in graph[user] and neigh != user:
                        suggestions.add(neigh)
        level += 1

    return sorted(suggestions)

def bellman_ford(edges: List[Tuple[str, str, float]], vertices: List[str], source: str):
    """
    Return dict of shortest distances from source to each vertex, or
    raise ValueError on negative-weight cycle detection.
    """
    dist = {v: float('inf') for v in vertices}
    dist[source] = 0.0

    for _ in range(len(vertices) - 1):
        updated = False
        for u, v, w in edges:
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                updated = True
        if not updated:
            break

    # check negative cycles
    for u, v, w in edges:
        if dist[u] != float('inf') and dist[u] + w < dist[v]:
            raise ValueError("Graph contains a negative-weight cycle")

    return dist

def dijkstra(graph: Dict[str, List[Tuple[str, float]]], source: str):
    """
    graph: adjacency list {node: [(neighbor, weight), ...]}
    returns: distances dict
    """
    dist = {node: float('inf') for node in graph}
    dist[source] = 0.0
    pq = [(0.0, source)]

    while pq:
        cur_d, node = heapq.heappop(pq)
        if cur_d > dist[node]:
            continue
        for neigh, w in graph[node]:
            nd = cur_d + w
            if nd < dist[neigh]:
                dist[neigh] = nd
                heapq.heappush(pq, (nd, neigh))

    return dist
def prim_mst(graph: Dict[str, List[Tuple[str, float]]], start: str = None):
    """
    Prim's algorithm using a min-heap.
    graph adjacency list: {node: [(neighbor, weight), ...], ...}
    returns (total_cost, list_of_edges_in_mst)
    """
    if not graph:
        return 0.0, []

    nodes = list(graph.keys())
    start = start or nodes[0]
    visited = set([start])
    edges_heap = []
    for v, w in graph[start]:
        heapq.heappush(edges_heap, (w, start, v))

    mst_edges = []
    total_cost = 0.0

    while edges_heap and len(visited) < len(nodes):
        w, u, v = heapq.heappop(edges_heap)
        if v in visited:
            continue
        visited.add(v)
        mst_edges.append((u, v, w))
        total_cost += w
        for to, wt in graph[v]:
            if to not in visited:
                heapq.heappush(edges_heap, (wt, v, to))

    if len(visited) != len(nodes):
        raise ValueError("Graph not connected - MST not possible for all nodes")

    return total_cost, mst_edges

class UnionFind:
    def __init__(self, elements):
        self.parent = {e: e for e in elements}
        self.rank = {e: 0 for e in elements}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1
        return True


def kruskal_mst(edge_list: List[Tuple[str, str, float]], vertices: List[str]):
    """
    edge_list: [(u, v, w), ...] undirected (u-v is same as v-u)
    vertices: list of nodes
    returns (total_cost, chosen_edges)
    """
    uf = UnionFind(vertices)
    sorted_edges = sorted(edge_list, key=lambda x: x[2])
    mst_edges = []
    total = 0.0
    for u, v, w in sorted_edges:
        if uf.union(u, v):
            mst_edges.append((u, v, w))
            total += w
    roots = set(uf.find(v) for v in vertices)
    if len(roots) > 1:
        raise ValueError("Graph not connected - MST not possible for all nodes")
    return total, mst_edges

def profile_function(func, *args, **kwargs):
    """
    Profiles time (seconds) and peak memory (MB) for a single function call.
    Returns: (result, elapsed_time_seconds, peak_memory_mb or None)
    Requires memory_profiler to be installed for memory measurement.
    """

    t0 = time.perf_counter()

    if memory_usage is not None:
      
        mem_samples, retval = memory_usage((func, args, kwargs), retval=True, interval=0.01, timeout=None)
        t1 = time.perf_counter()
        elapsed = t1 - t0
       
        peak = max(mem_samples) if mem_samples else None
        return retval, elapsed, peak
    else:
        
        retval = func(*args, **kwargs)
        t1 = time.perf_counter()
        elapsed = t1 - t0
        return retval, elapsed, None


def experimental_profile_plot():
    """
    Generate small experiment: measure run time for Prim, Kruskal, Dijkstra
    on random graphs of increasing size and plot time vs nodes.
    (This is optional and lightweight.)
    """
    if plt is None:
        print("matplotlib not installed. Skipping plotting.")
        return

    sizes = [50, 100, 200, 400]
    prim_times = []
    kruskal_times = []
    dijkstra_times = []

    for n in sizes:
       
        nodes = [f"N{i}" for i in range(n)]
        
        adj = {node: [] for node in nodes}
        edges = []
        for i in range(n - 1):
            w = random.uniform(1, 10)
            adj[nodes[i]].append((nodes[i + 1], w))
            adj[nodes[i + 1]].append((nodes[i], w))
            edges.append((nodes[i], nodes[i + 1], w))
     
        extra = max(1, n // 5)
        for _ in range(extra):
            u = random.choice(nodes)
            v = random.choice(nodes)
            if u == v:
                continue
            w = random.uniform(1, 10)
            adj[u].append((v, w))
            adj[v].append((u, w))
            edges.append((u, v, w))

        _, t_prim, _ = profile_function(lambda: prim_mst(adj))
        prim_times.append(t_prim)

       
        _, t_kruskal, _ = profile_function(lambda: kruskal_mst(edges, nodes))
        kruskal_times.append(t_kruskal)


        gdir = {node: [] for node in nodes}
        for u, v, w in edges:
            gdir[u].append((v, w))
            gdir[v].append((u, w))
        source = nodes[0]
        _, t_dij, _ = profile_function(lambda: dijkstra(gdir, source))
        dijkstra_times.append(t_dij)

        print(f"Done profiling for n={n}: prim={t_prim:.4f}s kruskal={t_kruskal:.4f}s dijkstra={t_dij:.4f}s")

    
    plt.figure(figsize=(8, 5))
    plt.plot(sizes, prim_times, label="Prim's")
    plt.plot(sizes, kruskal_times, label="Kruskal's")
    plt.plot(sizes, dijkstra_times, label="Dijkstra's")
    plt.xlabel("Number of nodes")
    plt.ylabel("Execution time (s)")
    plt.title("Algorithm runtime (small experiment)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def example_run_all():
    print("\n=== Problem 1: Friend Suggestion (BFS) ===")
    social_graph = {
        'A': ['B', 'C'],
        'B': ['A', 'D'],
        'C': ['A', 'D', 'E'],
        'D': ['B', 'C', 'F'],
        'E': ['C'],
        'F': ['D']
    }
    print("Suggested for A:", suggest_friends_bfs(social_graph, 'A'))

    print("\n=== Problem 2: Bellman-Ford ===")
    vertices = ['A', 'B', 'C', 'D', 'E']
    edges = [
        ('A', 'B', 4),
        ('A', 'C', 2),
        ('B', 'C', 3),
        ('B', 'D', 2),
        ('C', 'B', 1),
        ('C', 'D', 4),
        ('D', 'E', -5)
    ]
    try:
        dist = bellman_ford(edges, vertices, 'A')
        print("Distances from A:", dist)
    except ValueError as e:
        print("Bellman-Ford error:", e)

    print("\n=== Problem 3: Dijkstra ===")
    city_graph = {
        'A': [('B', 4), ('C', 2)],
        'B': [('A', 4), ('C', 5), ('D', 10)],
        'C': [('A', 2), ('B', 5), ('D', 3)],
        'D': [('B', 10), ('C', 3), ('E', 4)],
        'E': [('D', 4)]
    }
    dij = dijkstra(city_graph, 'A')
    print("Dijkstra distances from A:", dij)

    print("\n=== Problem 4: MST (Prim & Kruskal) ===")
   
    undirected_adj = {
        'A': [('B', 1), ('C', 3)],
        'B': [('A', 1), ('C', 1), ('D', 4)],
        'C': [('A', 3), ('B', 1), ('D', 2)],
        'D': [('B', 4), ('C', 2)]
    }
    prim_total, prim_edges = prim_mst(undirected_adj, start='A')
    print("Prim MST total cost:", prim_total)
    print("Prim MST edges:", prim_edges)

    
    edges = []
    seen = set()
    for u, neighs in undirected_adj.items():
        for v, w in neighs:
            key = tuple(sorted((u, v)))
            if key in seen:
                continue
            seen.add(key)
            edges.append((u, v, w))
    vertices = list(undirected_adj.keys())
    kruskal_total, kruskal_edges = kruskal_mst(edges, vertices)
    print("Kruskal MST total cost:", kruskal_total)
    print("Kruskal MST edges:", kruskal_edges)


def main_menu():
    while True:
        print("\n---------------- Graph Assignment Tools ----------------")
        print("1) Run example demos for all problems")
        print("2) Profile a single example (time+mem)")
        print("3) Run small experimental profiling & plot (optional)")
        print("4) Exit")
        choice = input("Choose (1-4): ").strip()
        if choice == '1':
            example_run_all()
        elif choice == '2':
            print("\nSelect which function to profile:")
            print(" a) Friend suggestion (BFS)")
            print(" b) Bellman-Ford")
            print(" c) Dijkstra")
            print(" d) Prim's MST")
            print(" e) Kruskal's MST")
            sub = input("Choose (a-e): ").strip().lower()
            if sub == 'a':
                g = {'A': ['B', 'C'], 'B': ['A', 'D'], 'C': ['A', 'D', 'E'], 'D': ['B', 'C'], 'E': ['C']}
                res, t, mem = profile_function(suggest_friends_bfs, g, 'A')
                print("Result:", res)
            elif sub == 'b':
                verts = ['A', 'B', 'C', 'D']
                eds = [('A', 'B', 1), ('B', 'C', -2), ('C', 'D', 2), ('A', 'D', 10)]
                try:
                    res, t, mem = profile_function(bellman_ford, eds, verts, 'A')
                    print("Result:", res)
                except Exception as e:
                    print("Error:", e)
                    res, t, mem = None, None, None
            elif sub == 'c':
                g = {'A': [('B', 5), ('C', 2)], 'B': [('A', 5), ('C', 1)], 'C': [('A', 2), ('B', 1)]}
                res, t, mem = profile_function(dijkstra, g, 'A')
                print("Result:", res)
            elif sub == 'd':
                adj = {'A': [('B', 1)], 'B': [('A', 1), ('C', 2)], 'C': [('B', 2)]}
                res, t, mem = profile_function(lambda: prim_mst(adj))
                print("Result:", res)
            elif sub == 'e':
                verts = ['A', 'B', 'C']
                eds = [('A', 'B', 1), ('B', 'C', 2), ('A', 'C', 3)]
                res, t, mem = profile_function(lambda: kruskal_mst(eds, verts))
                print("Result:", res)
            else:
                print("Invalid choice.")
                continue
            print(f"Time elapsed: {t:.6f} s")
            if mem is not None:
                print(f"Peak memory (MB): {mem:.3f}")
            else:
                print("Memory profiling not available (install memory_profiler).")
        elif choice == '3':
            if plt is None:
                print("matplotlib not installed. Install it to use plotting.")
            else:
                experimental_profile_plot()
        elif choice == '4':
            print("Bye.")
            break
        else:
            print("Invalid input, try again.")


if __name__ == "__main__":
    print("Graph Real-World Assignment Tools")
    print(" - Make sure to install dependencies listed at the top of the file.")
    print(" - For memory profiling, install `memory_profiler` (pip install memory_profiler).")
    main_menu()

