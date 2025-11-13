Graph = {
  'A': ['B', 'C'],
  'B': ['A', 'D'],
  'C': ['A', 'D', 'E'],
  'D': ['B', 'C'],
  'E': ['C']
}
from collections import deque

def suggest_friends(graph, user):
    visited = set()
    queue = deque([user])
    visited.add(user)
    
    direct_friends = set(graph[user])
    suggestions = set()
    
    while queue:
        current = queue.popleft()
        for neighbor in graph[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                
                if neighbor not in direct_friends and neighbor != user:
                    suggestions.add(neighbor)
    
    return list(suggestions)

graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D'],
    'C': ['A', 'D', 'E'],
    'D': ['B', 'C'],
    'E': ['C']
}

print("Friend suggestions for A:", suggest_friends(graph, 'A'))
