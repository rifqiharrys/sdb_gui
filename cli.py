from pathlib import Path

import typer

from sdb.io import read_geotiff, read_shapefile, write_geotiff, write_shapefile
from sdb.preprocessing import reproject_vector
from sdb.utils import median_filter, point_sampling

cli_app = typer.Typer()

@cli_app.command(name="median-filter")
def median_filter_cli(
    input_file: str,
    output_file: str,
    filter_size: int = 3
) -> None:
    """
    Apply median filter to a raster image.

    Parameters
    ----------
    input_file : str
        Path to the input raster file.
    output_file : str
        Path to the output raster file.
    filter_size : int, optional
        Size of the median filter window. Must be >= 3 and odd. Default is 3.

    Returns
    -------
    None
    """

    raster = read_geotiff(Path(input_file))

    filtered_array = median_filter(raster.values, filter_size=filter_size)
    filtered_raster = raster.copy(data=filtered_array)

    write_geotiff(
        raster=filtered_raster,
        raster_loc=Path(output_file),
        to_tif=True
    )


@cli_app.command(name="point-sampling")
def point_sampling_cli(
    input_raster: str,
    input_vector: str,
    output_vector: str
)-> None:
    """
    Sample raster values at point locations and save the results to a new vector file.

    Parameters
    ----------
    input_raster : str
        Path to the input raster file.
    input_vector : str
        Path to the input vector file containing point locations.
    output_vector : str
        Path to the output vector file where sampled results will be saved.

    Returns
    -------
    None
    """

    raster = read_geotiff(Path(input_raster))
    vector = read_shapefile(Path(input_vector))

    # check if the vector data contains point geometry
    if not all(vector.geometry.type == 'Point'):
        raise ValueError('Input vector data must contain point geometry')
    
    # check if the raster and vector data have the same CRS
    # reproject the vector data to the raster CRS if they are different
    if raster.rio.crs != vector.crs:
        vector = reproject_vector(
            raster=raster,
            vector=vector
        )

    sampled_table = point_sampling(
        raster=raster,
        x=vector.geometry.x,
        y=vector.geometry.y
    )

    write_shapefile(
        table=sampled_table,
        vector_loc=Path(output_vector),
        x_col_name='x',
        y_col_name='y',
        crs=raster.rio.crs
    )


if __name__ == "__main__":
    cli_app()