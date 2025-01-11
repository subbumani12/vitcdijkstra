#include <stdio.h>
#include <limits.h>

#define INF INT_MAX
#define MAX_NODES 100

int findMinDistanceNode(int distance[], int visited[], int v) {
    int min = INF, min_index = -1;
    for (int i = 0; i < v; i++) {
        if (!visited[i] && distance[i] < min) {
            min = distance[i];
            min_index = i;
        }
    }
    return min_index;
}

int* DijkstraShortestPath(int graph[MAX_NODES][MAX_NODES], int source, int dst, int v) {
    int distance[v];       
    int visited[v];        
    int parent[v];         
    for (int i = 0; i < v; i++) {
        distance[i] = INF;
        visited[i] = 0;
        parent[i] = -1;
    }
    distance[source] = 0; 
    parent[source] = source;
    for (int count = 0; count < v - 1; count++) {
        int u = findMinDistanceNode(distance, visited, v); 
        if (u == -1 || u == dst) break;                   
        visited[u] = 1;                                   
        for (int i = 0; i < v; i++) {
            if (graph[u][i] != 0 && !visited[i] && distance[u] != INF && 
                distance[u] + graph[u][i] < distance[i]) {
                distance[i] = distance[u] + graph[u][i];
                parent[i] = u;
            }
        }
    }
    return distance;
}
