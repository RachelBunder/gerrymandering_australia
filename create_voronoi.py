import pandas as pd
import geopandas as gpd
import numpy as np
import shapely
import os
import json

import plotly.express as px
import plotly.graph_objects as go

from scipy.spatial import Voronoi
import plotly.express as px
import numpy as np


from shapely.ops import unary_union
from shapely.geometry import Polygon


def get_clean_booths(
    booth_loc: str = "data/20190518/GeneralPollingPlacesDownload-24310.csv",
    booth_url: str = "https://results.aec.gov.au/24310/Website/Downloads/GeneralPollingPlacesDownload-24310.csv",
) -> gpd.GeoDataFrame:
    """Load and filter polling booth data, downloading from the AEC if not found locally.

    Parameters
    ----------
    booth_loc : str
        Path to the local CSV file containing polling place data.

    Returns
    -------
    gpd.GeoDataFrame
        DataFrame containing only standard polling places (PollingPlaceTypeID == 1).
    """

    try:
        booths = pd.read_csv(booth_loc, skiprows=1, dtype={"PollingPlaceID": str})
    except IOError:
        booths = gpd.read_file(booth_url)
        booths.to_csv(booth_loc)
        booths = pd.read_csv(booth_loc, skiprows=1)

    booths = booths[booths["PollingPlaceTypeID"] == 1]
    booths = gpd.GeoDataFrame(booths)
    return booths


def create_voronoi(
    booths: gpd.GeoDataFrame,
    state: str,
    state_boundary_url="https://data.gov.au/data/dataset/bdcf5b09-89bc-47ec-9281-6b8e9ee147aa/resource/3e45fe50-aaf8-48e7-a78a-e2497ff84372/download/aug20_adminbounds_esrishapefileordbffile_gda2020.zip",
    voronoi_radius: int = 2000000,
    crs_code: str = "EPSG:3033",
) -> gpd.GeoSeries:
    """Generate Voronoi polygons for polling booths within a state boundary.

    Projects booth coordinates to a metric CRS, computes Voronoi tessellation,
    then clips each polygon to the state's electoral boundary and reprojects to
    WGS84.

    Parameters
    ----------
    booths : gpd.GeoDataFrame
        DataFrame of polling booths with 'Longitude', 'Latitude', and
        'PollingPlaceID' columns.
    state : str
        Two- or three-letter state abbreviation used to locate the matching
        boundary shapefile (e.g. 'NSW', 'VIC').

    Returns
    -------
    gpd.GeoSeries
        Clipped Voronoi polygons in EPSG:4326, indexed by PollingPlaceID.
    """
    # state_boundary_file=f"data/borders/AUG20_AdminBounds_ESRIShapefileorDBFfile_GDA2020/Administrative Boundaries/State Electoral Boundaries AUGUST 2020/Standard/{state}_STATE_ELECTORAL_POLYGON_shp.shp",
    state_boundary_file = f"data/borders/AUG20_AdminBounds_ESRIShapefileorDBFfile_GDA2020/Administrative Boundaries/State Electoral Boundaries AUGUST 2020/Standard/{state}_STATE_ELECTORAL_POLYGON_shp.shp"
    booths = gpd.GeoDataFrame(booths)
    booths["geometry"] = gpd.points_from_xy(
        x=booths["Longitude"], y=booths["Latitude"], crs="EPSG:4326"
    )
    booths = booths.to_crs(crs_code)
    booths["id"] = booths.reset_index(drop=True).index.values

    coords = [(c.x, c.y) for c in booths["geometry"]]

    vor = Voronoi(coords)
    regions, vertices = voronoi_finite_polygons_2d(vor, voronoi_radius)

    # Clip by state border
    try:
        border = gpd.read_file(state_boundary_file)
    except IOError:
        from urllib.request import urlopen
        from tempfile import NamedTemporaryFile
        from shutil import unpack_archive

        with urlopen(state_boundary_url) as zipresp, NamedTemporaryFile() as tfile:
            tfile.write(zipresp.read())
            tfile.seek(0)
            unpack_archive(tfile.name, "data/borders/", format="zip")

            border = gpd.read_file(state_boundary_file)

    # Converting back to round
    vertices_t = gpd.GeoSeries(
        gpd.points_from_xy(
            [v[0] for v in vertices], [v[1] for v in vertices], crs=crs_code
        )
    )
    vertices_t = vertices_t.to_crs("EPSG:4326")
    vertices = [(v.x, v.y) for v in vertices_t.values]

    border_p = unary_union(border["geometry"])

    # trim the vertices so that they don't go past stateorder
    polygons = [
        Polygon([vertices[v] for v in region]).buffer(0).intersection(border_p)
        for region in regions
    ]

    return gpd.GeoSeries(polygons, index=booths["PollingPlaceID"], crs=crs_code)


def voronoi_finite_polygons_2d(
    vor: Voronoi, radius: float | None = None
) -> tuple[list, np.ndarray]:
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions. Taken from https://stackoverflow.com/questions/57385472/how-to-set-a-fixed-outer-boundary-to-voronoi-tessellations
    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.
    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.
    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()
    new_point_region = []

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max() * 2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            tangent = vor.points[p2] - vor.points[p1]  # tangent
            tangent = tangent / np.linalg.norm(tangent)
            normal = np.array([-tangent[1], tangent[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, normal)) * normal
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def plot_voronoi(booths):
    min_lon, min_lat, max_lon, max_lat = booths["voronoi"].total_bounds
    centre_lon = (max_lon + min_lon) / 2
    centre_lat = (min_lat + max_lat) / 2
    area = (max_lon - min_lon) * (max_lat - min_lat) * 10
    num_regions = len(booths)
    zoom = np.interp(
        x=area,
        xp=[0, 5**-10, 4**-10, 3**-10, 2**-10, 0.0025, 1**-10, 2**10, 45324, 5**10],
        fp=[20, 17, 16, 15, 14, 11, 7, 5, 3, 1],
    )

    booth_json = json.loads(booths["voronoi"].to_json(show_bbox=False))

    fig = px.choropleth_map(
        booths,
        geojson=booth_json,
        locations="PollingPlaceID",
        color="PollingPlaceID",
        color_continuous_scale="Viridis",
        range_color=(0, num_regions),
        map_style="carto-positron",
        hover_data=["PremisesNm", "Longitude", "Latitude"],
        opacity=0.5,
        labels={"PollingPlaceID": "Input ID"},
        center={"lat": centre_lat, "lon": centre_lon},
        zoom=zoom,
    )

    fig.add_traces(
        px.scatter_map(
            booths,
            lat="Latitude",
            lon="Longitude",
            hover_name="PremisesNm",
            hover_data=["DivisionNm", "PollingPlaceNm", "PollingPlaceID"],
        ).data
    )

    fig.update_layout(showlegend=False)
    return fig


if __name__ == "__main__":
    # Get booths: just return the ones we want
    booths = get_clean_booths()

    booths.to_csv("tmp_check_booth_index.csv")

    booths = booths.set_index("PollingPlaceID", drop=False)

    for state in booths["State"].unique():
        print(state)
        state_booth = booths[booths["State"] == state]
        state_booth["geometry"] = create_voronoi(
            state_booth, state
        )  # .set_index('PollingPlaceID', drop=True)
        # print(tmp.index)
        # print(booths.index)
        # gpd.GeoDataFrame(tmp, crs='epsg:4326').to_file('tmp_check_groupby_index.geojson', driver='GeoJSON')
        print(state)
        state_booth = state_booth.drop("PollingPlaceID", axis=1)
        state_booth = gpd.GeoDataFrame(state_booth, crs="epsg:4326")
        state_booth.to_file(f"data/20190518/voronoi_{state}.geojson", driver="GeoJSON")
