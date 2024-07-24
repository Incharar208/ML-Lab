from queue import PriorityQueue

def bestFirstSearch(graph, start, goal, heuristic):
    visited = set()
    pq = PriorityQueue()
    pq.put((heuristic[start], start))

    while not pq.empty():
        h, node = pq.get()
        if node == goal:
            print(node)
            return
        
        if node not in visited:
            print(node, "->", end = " ")
            visited.add(node)

            for neighbor, i in graph[node]:
                if neighbor not in visited:
                    pq.put((heuristic[neighbor], neighbor))
        
    print("Goal not found!")

graph = {
    'A': [('B', 11), ('C', 14), ('D', 7)],
    'B': [('A', 11), ('E', 15)],
    'C': [('A', 14), ('E', 8), ('D', 18), ('F', 10)],
    'D': [('A', 7), ('F', 25), ('C', 18)],
    'E': [('B', 15), ('C', 8), ('H', 9)],
    'F': [('G', 20), ('C', 10), ('D', 25)],
    'G': [],
    'H': [('E', 9), ('G', 10)]
}

start = 'A'
goal = 'G'

heuristic = {
    'A': 40,
    'B': 32,
    'C': 25,
    'D': 35,
    'E': 19,
    'F': 17,
    'G': 0,
    'H': 10
}

bestFirstSearch(graph, start, goal, heuristic)