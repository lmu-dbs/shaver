from colour import Color


def get_color(number: float):
    corresponding_value = min(COLOR_SCALE.values(), key=lambda x: abs(x - number))
    return list(COLOR_SCALE.keys())[list(COLOR_SCALE.values()).index(corresponding_value)]


def expand_hex_color(hex_color):
    expanded_color = hex_color
    # Remove the leading "#" if present
    if hex_color.startswith("#"):
        hex_color = hex_color[1:]
        expanded_color = "#" + hex_color
    # Expand the hex color abbreviation
    if len(hex_color) == 3 and hex_color != "red":
        # Convert "#RGB" to "#RRGGBB"
        hex_color = ''.join([ch * 2 for ch in hex_color])
        expanded_color = "#" + hex_color
    # Add the leading "#" back to the expanded color
    return expanded_color


def get_color_range(from_color, to_color, n: int):
    blue = Color(from_color)
    red = Color(to_color)

    color_list = list(blue.range_to(red, n))
    color_list = [expand_hex_color(x.__str__()) for x in color_list]
    color_dict = dict()
    for i in range(n):
        color_dict[color_list[i]] = (i + 1) / n
    return color_dict


def colorize_graph(normalized_r1, graph, mapping_dict):
    relevant_nodes = []
    for node, shapleys in normalized_r1.items():
        for n in graph.get_nodes():
            if n.obj_dict["attributes"]["label"].startswith(mapping_dict[str(node)]):
                n.obj_dict["attributes"]["fillcolor"] = get_color(normalized_r1[node])
                n.obj_dict["attributes"]["label"] = mapping_dict[str(node)]
                relevant_nodes.append(n.get_name())
                break
    for x in graph.get_nodes():
        if x.get_name() not in relevant_nodes:
            x.obj_dict["attributes"]["fillcolor"] = '#ffffff'
        if x.obj_dict["attributes"]["label"] == "@@S":
            x.obj_dict["attributes"]["fillcolor"] = '#2ACB2A'
        if x.obj_dict["attributes"]["label"] == "@@E":
            x.obj_dict["attributes"]["fillcolor"] = '#FFA604'
    for x in graph.get_edges():
        x.obj_dict["attributes"]["label"] = ""


COLOR_SCALE = get_color_range(from_color="white", to_color="red", n=50)
