import heapq 

def dijkstra(graph, source):
    distances = {node: float('inf') for node in graph}
    distances[source] = 0
    pq = [(0, source)]

    while pq:
        current_distance, current_node = heapq.heappop(pq)

        if current_distance > distances[current_node]:
            continue

        for neighbor, weight in graph[current_node]:
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))

    return distances
city_graph = {
    'A': [('B', 4), ('C', 2)],
    'B': [('A', 4), ('C', 5), ('D', 10)],
    'C': [('A', 2), ('B', 5), ('D', 3)],
    'D': [('B', 10), ('C', 3), ('E', 4)],
    'E': [('D', 4)]
}

source = 'A'
print("====================================")
print("Problem 3: Emergency Response System")
print("====================================")
distances = dijkstra(city_graph, source)

print(f"Fastest routes (in minutes) from {source}:")
for node in distances:
    print(f"{node}: {distances[node]}")
    
print("\n--- Analysis ---")
print("• Time Complexity: O(E log V) using a min-heap.")
print("• Dijkstra’s algorithm is unsuitable for graphs with negative weights,")
print("  because it assumes once a node’s shortest distance is found, it cannot be improved later.")

