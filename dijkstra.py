import plotly.graph_objects as go
import osmnx as ox
import numpy as np
import webbrowser
import xmltodict as xtd
import os
import sys
import folium
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Parsing bounds from .OSM file
def parse_osm_bounds(file_path):
    with open(file_path, "rb") as osm_fn:
        map_osm = xtd.parse(osm_fn)['osm']
    ymax = float(map_osm['bounds']['@maxlat'])
    ymin = float(map_osm['bounds']['@minlat'])
    xmax = float(map_osm['bounds']['@maxlon'])
    xmin = float(map_osm['bounds']['@minlon'])
    return ymax, ymin, xmax, xmin

class Node_Distance:
    def __init__(self, name, dist):
        self.name = name
        self.dist = dist

class Graph:
    def __init__(self, node_count):
        self.adjlist = defaultdict(list)
        self.node_count = node_count
        self.node_coordinates = {}  # Add storage for node coordinates

    def Add_Into_Adjlist(self, src, node_dist):
        self.adjlist[src].append(node_dist)

    def Add_Node_Coordinates(self, node_id, lat, lon):
        self.node_coordinates[node_id] = (lat, lon)

    def Dijkstras_Shortest_Path(self, source, dst, v):
        distance = [float('inf')] * self.node_count
        distance[source] = 0
        parent = list(range(v))
        dict_node_length = {source: 0}

        while dict_node_length:
            current_source_node = min(dict_node_length, key=dict_node_length.get)
            del dict_node_length[current_source_node]
            if current_source_node == dst:
                break

            for node_dist in self.adjlist[current_source_node]:
                adjnode = node_dist.name
                length_to_adjnode = node_dist.dist

                if distance[adjnode] > distance[current_source_node] + length_to_adjnode:
                    parent[adjnode] = current_source_node
                    distance[adjnode] = distance[current_source_node] + length_to_adjnode
                    dict_node_length[adjnode] = distance[adjnode]
        return distance[dst], parent

def Setup(north, east, south, west):
    G = ox.graph_from_bbox(north=north, south=south, east=east, west=west, network_type='all')
    v = len(G.nodes)
    di = {}
    index = 0
    
    # Create graph instance
    graph = Graph(v)
    
    # Store node coordinates
    for node in G.nodes(data=True):
        node_id = node[0]
        di[node_id] = [index]
        di[index] = [node_id]
        # Store coordinates in the graph
        graph.Add_Node_Coordinates(index, G.nodes[node_id]['y'], G.nodes[node_id]['x'])
        index += 1

    li = []
    for edge in G.edges(data=True):
        src_id = edge[0]
        dst_id = edge[1]
        new_src = di[src_id][0]
        new_dst = di[dst_id][0]
        weight = edge[2]['length']
        li.append([new_src, new_dst, weight])

    ox.plot_graph(G, edge_color="y", save=True, filepath="MapImage.jpg", show=True)
    e = len(li)
    
    # Add edges to graph
    for i in range(e):
        graph.Add_Into_Adjlist(li[i][0], Node_Distance(li[i][1], li[i][2]))
    
    return G, graph, v, e, li, di

def BuildAllNodesMap(graph, bounds):
    """
    Create an interactive Folium map showing all nodes with their coordinates
    
    Args:
        graph: Graph instance containing node coordinates
        bounds: Tuple of (north, east, south, west) coordinates
    """
    # Calculate center point
    north, east, south, west = bounds
    center_lat = (north + south) / 2
    center_lon = (east + west) / 2
    
    # Create base map
    map_0 = folium.Map(location=[center_lat, center_lon], zoom_start=16)
    
    # Add markers for all nodes
    for node_id, coords in graph.node_coordinates.items():
        lat, lon = coords
        popup_text = f'Node {node_id}'
        
        folium.CircleMarker(
            location=[lat, lon],
            radius=3,
            color="green",
            fill=True,
            fill_color="green",
            popup=popup_text
        ).add_to(map_0)
    
    # Save the map
    map_0.save('interactive_nodes.html')
    return map_0

def getPath(parent, dst, src, di):
    path = []
    current = dst
    while current != src:
        path.append(di[current][0])
        current = parent[current]
    path.append(di[src][0])
    path.reverse()
    return path

def node_list_to_path(G, node_list):
    edge_nodes = list(zip(node_list[:-1], node_list[1:]))
    lines = []
    for u, v in edge_nodes:
        data = min(G.get_edge_data(u, v).values(), key=lambda x: x['length'])
        if 'geometry' in data:
            xs, ys = data['geometry'].xy
            lines.append(list(zip(xs, ys)))
        else:
            x1 = G.nodes[u]['x']
            y1 = G.nodes[u]['y']
            x2 = G.nodes[v]['x']
            y2 = G.nodes[v]['y']
            line = [(x1, y1), (x2, y2)]
            lines.append(line)
    return lines

def calc_lat_long(G, path):
    lines = node_list_to_path(G, path)
    long2, lat2 = [], []
    for line in lines:
        for coord in line:
            long2.append(coord[0])
            lat2.append(coord[1])
    return lat2, long2
def plot_path(lat, long, origin_point, destination_point, ymax, xmax, ymin, xmin, distance=None):
    # Calculate time if distance is provided
    if distance is not None:
        distance_km = distance / 1000
        time_hours = distance_km / 7.5
        time_minutes = time_hours * 60
        minutes = int(time_minutes)
        seconds = int((time_minutes - minutes) * 60)

    fig = go.Figure(go.Scattermapbox(
        name="Path",
        mode="lines",
        lon=long,
        lat=lat,
        marker={'size': 10},
        line=dict(width=4.5, color='blue')
    ))

    fig.add_trace(go.Scattermapbox(
        name="Source", 
        mode="markers",
        lon=[origin_point[1]], 
        lat=[origin_point[0]],
        marker={'size': 12, 'color': "red"}
    ))

    fig.add_trace(go.Scattermapbox(
        name="Destination",
        mode="markers", 
        lon=[destination_point[1]],
        lat=[destination_point[0]],
        marker={'size': 12, 'color': 'green'}
    ))

    # Create annotations for distance and time
    annotations = []
    if distance is not None:
        if minutes == 0:
            time_text = f"{seconds} seconds"
        else:
            time_text = f"{minutes} min {seconds} sec"
            
        legend_height = 0.08
        box_height = 0.06
        gap_to_legend = 0.06           
        gap_to_time = 0.03           
        
        pos_1 = 0.99 - (legend_height + gap_to_legend * 3)    
        pos_2 = pos_1 - (box_height + gap_to_time)            
            
        annotations.extend([
            dict(
                x=0.99,
                y=pos_1,
                xref="paper",
                yref="paper",
                text=f"Distance: {distance:.2f} meters",
                showarrow=False,
                font=dict(size=12),
                align="right",
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="black",
                borderwidth=1,
                borderpad=4,
                xanchor="right"
            ),
            dict(
                x=0.99,
                y=pos_2,
                xref="paper",
                yref="paper",
                text=f"Estimated Walking Time: {time_text}",
                showarrow=False,
                font=dict(size=12),
                align="right",
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="black",
                borderwidth=1,
                borderpad=4,
                xanchor="right"
            )
        ])
    
    fig.update_layout(
        mapbox_style="open-street-map",
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        mapbox={
            'center': {'lat': np.mean(lat), 'lon': np.mean(long)},
            'bounds': {
                'north': ymax,
                'south': ymin,
                'east': xmax, 
                'west': xmin
            }
        },
        showlegend=True,
        legend={
            'yanchor': "top",
            'y': 0.90,
            'xanchor': "right",
            'x': 0.99,
            'bgcolor': 'rgba(255, 255, 255, 0.8)',
            'bordercolor': 'black',
            'borderwidth': 1
        },
        annotations=annotations
    )

    fig.write_html("OutputMap.html")
    OpenHTMLMapinBrowser("OutputMap.html")

def OpenHTMLMapinBrowser(filename):
    url = "file://" + os.path.realpath(filename)
    webbrowser.open(url, new=2)

def main():
    print("Map Navigation Program\n")

    # Parse OSM file bounds
    try:
        ymax, ymin, xmax, xmin = parse_osm_bounds("Maps/VIT_Chennai.osm")
        print(f"\nMap Bounds:")
        print(f"North: {ymax}, South: {ymin}")
        print(f"East: {xmax}, West: {xmin}")
    except FileNotFoundError:
        print("Error: Maps/mapHSR.osm file not found!")
        return
    except Exception as e:
        print(f"Error parsing OSM file: {e}")
        return

    # Setup graph using OSM bounds
    bounds = (ymax, xmax, ymin, xmin)  # north, east, south, west
    G, graph, v, e, li, di = Setup(*bounds)
    
    # Create and display interactive node map
    interactive_map = BuildAllNodesMap(graph, bounds)
    print("\nInteractive node map saved as interactive_nodes.html")
    print("Opening map in browser - please look at the node numbers...")
    OpenHTMLMapinBrowser("interactive_nodes.html")
    
    print("\nGraph image saved as MapImage.jpg")

    # Get user input for source and destination nodes
    while True:
        while True:
            try:
                source_node = int(input("\nEnter source node number (0 to {}): ".format(v-1)))
                if 0 <= source_node < v:
                    break
                print(f"Please enter a number between 0 and {v-1}")
            except ValueError:
                print("Please enter a valid number")
        
        while True:
            try:
                dest_node = int(input("Enter destination node number (0 to {}): ".format(v-1)))
                if 0 <= dest_node < v:
                    if dest_node != source_node:
                        break
                    print("Destination node must be different from source node")
                else:
                    print(f"Please enter a number between 0 and {v-1}")
            except ValueError:
                print("Please enter a valid number")

        # Get coordinates for source and destination nodes
        src_coords = graph.node_coordinates[source_node]
        dst_coords = graph.node_coordinates[dest_node]
        
        print(f"\nSource node coordinates: {src_coords}")
        print(f"Destination node coordinates: {dst_coords}")

        # Calculate shortest path
        ShortestDist, parent = graph.Dijkstras_Shortest_Path(source_node, dest_node, v)

        # Get path and plot it
        path = getPath(parent, dest_node, source_node, di)
        lat2, long2 = calc_lat_long(G, path)
        
        print("Generating path visualization...")
        
        # Plot the path using the node coordinates
        plot_path(lat2, long2, src_coords, dst_coords, ymax, xmax, ymin, xmin, ShortestDist)
        print("\nPath map saved and opened as OutputMap.html")

        user = input("Enter 1 to continue: ")
        if user!="1":
            print("Program terminated")
            break

if __name__ == "__main__":
    main()
