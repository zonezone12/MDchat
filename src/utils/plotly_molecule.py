import numpy as np
import plotly.graph_objects as go

VDW_RADII = {
    "H": 0.32,
    "C": 0.75,
    "N": 0.71,
    "O": 0.63,
    "S": 1.03,
    "P": 1.11,
    "F": 0.64,
    "CL": 0.99,
    "BR": 1.14,
    "I": 1.33
}

def make_molecule_components(coords, elements,vdw_radii=VDW_RADII):
    """
    Build Plotly components (traces, annotations, menus) for a molecule.

    Parameters
    ----------
    coords : (N, 3) ndarray
        Cartesian coordinates in Å.
    elements : list[str] length N
        Element symbols (e.g. ['C', 'H', 'O', ...])

    Returns
    -------
    atom_trace : go.Scatter3d
    bond_trace : go.Scatter3d
    annotations_id : list[dict]
        Annotations with atom indices.
    annotations_length : list[dict]
        Annotations with bond lengths.
    updatemenus : list[dict]
        Plotly updatemenus to toggle annotation sets.
    """
    # Your radii and colors
    cpk_colors = dict(C='black', F='green', H='white', N='blue', O='red', S='yellow', P='orange')
    coords = np.asarray(coords, dtype=float)
    x_coordinates = coords[:, 0]
    y_coordinates = coords[:, 1]
    z_coordinates = coords[:, 2]

    # Radii with fallback to C-like
    radii = [vdw_radii.get(el, 0.77) for el in elements]

    # --- bond finder (your logic, generalized) ---
    def get_bonds():
        """Generates a dict of bonds: {frozenset({i, j}): distance}."""
        ids = np.arange(coords.shape[0])
        bonds = dict()

        coordinates_compare = coords.copy()
        radii_compare = np.array(radii, dtype=float).copy()
        ids_compare = ids.copy()

        for _ in range(len(ids)):
            coordinates_compare = np.roll(coordinates_compare, -1, axis=0)
            radii_compare = np.roll(radii_compare, -1, axis=0)
            ids_compare = np.roll(ids_compare, -1, axis=0)

            distances = np.linalg.norm(coords - coordinates_compare, axis=1)
            bond_distances = (np.array(radii) + radii_compare) * 1.3
            mask = np.logical_and(distances > 0.1, distances < bond_distances)
            distances = distances.round(2)

            new_bonds = {
                frozenset([i, j]): dist
                for i, j, dist in zip(ids[mask], ids_compare[mask], distances[mask])
            }
            bonds.update(new_bonds)

        return bonds

    bonds = get_bonds()

    # --- atom trace (your atom_trace) ---
    colors = [cpk_colors.get(el, 'grey') for el in elements]
    markers = dict(
        color=colors,
        line=dict(color='lightgray', width=2),
        size=5,
        symbol='circle',
        opacity=0.8,
    )
    atom_trace = go.Scatter3d(
        x=x_coordinates,
        y=y_coordinates,
        z=z_coordinates,
        mode='markers',
        marker=markers,
        text=elements,
        name='Atoms',
    )

    # --- bond trace (your bond_trace) ---
    bond_trace = go.Scatter3d(
        x=[],
        y=[],
        z=[],
        hoverinfo='none',
        mode='lines',
        marker=dict(color='grey', size=10, opacity=1),
        name='Bonds',
    )
    for i, j in bonds.keys():
        bond_trace['x'] += (x_coordinates[i], x_coordinates[j], None)
        bond_trace['y'] += (y_coordinates[i], y_coordinates[j], None)
        bond_trace['z'] += (z_coordinates[i], z_coordinates[j], None)

    # --- annotations: atom indices ---
    zipped = zip(range(len(elements)), x_coordinates, y_coordinates, z_coordinates)
    annotations_id = [
        dict(
            text=str(num),
            x=x,
            y=y,
            z=z,
            showarrow=False,
            yshift=15,
            font=dict(color="blue"),
        )
        for num, x, y, z in zipped
    ]

    # --- annotations: bond lengths ---
    annotations_length = []
    for (i, j), dist in bonds.items():
        x_middle, y_middle, z_middle = (coords[i] + coords[j]) / 2.0
        annotation = dict(
            text=str(dist),
            x=x_middle,
            y=y_middle,
            z=z_middle,
            showarrow=False,
            yshift=15,
        )
        annotations_length.append(annotation)

    # --- updatemenus (your menu logic) ---
    updatemenus = [
        dict(
            buttons=list(
                [
                    dict(
                        label='Atom indices',
                        method='relayout',
                        args=[{'scene.annotations': annotations_id}],
                    ),
                    dict(
                        label='Bond lengths',
                        method='relayout',
                        args=[{'scene.annotations': annotations_length}],
                    ),
                    dict(
                        label='Atom indices & Bond lengths',
                        method='relayout',
                        args=[{'scene.annotations': annotations_id + annotations_length}],
                    ),
                    dict(
                        label='Hide all',
                        method='relayout',
                        args=[{'scene.annotations': []}],
                    ),
                ]
            ),
            direction='down',
            xanchor='left',
            yanchor='top',
            x=0.0,
            y=1.0,
        )
    ]

    return atom_trace, bond_trace, annotations_id, annotations_length, updatemenus


def plot_molecule(coords, elements):
    """
    Standalone version of your original plot_molecule, but:
    - takes coords + elements directly
    - returns a Plotly Figure (no iplot inside)

    Parameters
    ----------
    coords : (N, 3) ndarray
    elements : list[str]

    Returns
    -------
    fig : go.Figure
    """
    atom_trace, bond_trace, annotations_id, annotations_length, updatemenus = \
        make_molecule_components(coords, elements)

    axis_params = dict(
        showgrid=False,
        showbackground=False,
        showticklabels=False,
        zeroline=False,
        titlefont=dict(color='white'),
    )

    layout = dict(
        scene=dict(
            xaxis=axis_params,
            yaxis=axis_params,
            zaxis=axis_params,
            annotations=annotations_id,
        ),
        margin=dict(r=0, l=0, b=0, t=0),
        showlegend=False,
        updatemenus=updatemenus,
    )

    fig = go.Figure(data=[atom_trace, bond_trace], layout=layout)
    return fig
