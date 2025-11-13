from collections import deque

def suggest_friends(graph, user):
    visited = set()
    queue = deque([user])
    visited.add(user)
    level = 0
    suggestions = set()

    while queue and level < 2:
        for _ in range(len(queue)):
            current = queue.popleft()
            for neighbor in graph[current]:
                if neighbor not in visited:
                    queue.append(neighbor)
                    visited.add(neighbor)
                    if level == 1 and neighbor not in graph[user]:
                        suggestions.add(neighbor)
        level += 1
    return list(suggestions)
    
social_graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D'],
    'C': ['A', 'D', 'E'],
    'D': ['B', 'C', 'F'],
    'E': ['C'],
    'F': ['D']
}
user = 'A'
print("====================================")
print("Problem 1: Friend Suggestion System")
print("====================================")
print(f"Suggested friends for {user}: {suggest_friends(social_graph, user)}\n")

def bellman_ford(edges, vertices, source):
    distance = {v: float('inf') for v in vertices}
    distance[source] = 0

    for _ in range(len(vertices) - 1):
        for u, v, w in edges:
            if distance[u] + w < distance[v]:
                distance[v] = distance[u] + w

    for u, v, w in edges:
        if distance[u] + w < distance[v]:
            print("Graph contains a negative weight cycle.")
            return None

    return distance

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
source = 'A'

print("====================================")
print("Problem 2: Route Finding (Bellman-Ford)")
print("====================================")
distances = bellman_ford(edges, vertices, source)
if distances:
    print(f"Shortest distances from {source}:")
    for v in vertices:
        print(f"{v}: {distances[v]}")

