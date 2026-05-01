from pathlib import Path
from typing import Annotated

import typer

from sdb.io import read_geotiff, read_shapefile, write_geotiff, write_shapefile
from sdb.utils import median_filter, point_sampling

cli_app = typer.Typer(no_args_is_help=True)

@cli_app.command(
    name='median-filter',
    no_args_is_help=True,
)
def median_filter_cli(
    input_file: Annotated[str, typer.Argument(
        help='Path to the input raster file.'
    )],
    output_file: Annotated[str, typer.Argument(
        help='Path to the output raster file.'
    )],
    filter_size: Annotated[int, typer.Option(
        '--filter-size',
        '-f',
        help='Size of the median filter window. '
        'Must be >= 3 and odd.'
    )] = 3
) -> None:
    """
    Apply median filter to a raster image.
    """

    raster = read_geotiff(Path(input_file))

    filtered_array = median_filter(raster.values, filter_size=filter_size)
    filtered_raster = raster.copy(data=filtered_array)

    write_geotiff(
        raster=filtered_raster,
        raster_loc=Path(output_file),
        to_tif=True
    )


@cli_app.command(
    name='point-sampling',
    no_args_is_help=True,
)
def point_sampling_cli(
    input_raster: Annotated[str, typer.Argument(
        help='Path to the input raster file.'
    )],
    input_vector: Annotated[str, typer.Argument(
        help='Path to the input vector file containing point locations.'
    )],
    output_vector: Annotated[str, typer.Argument(
        help='Path to the output vector file where sampled results will be saved.'
    )],
    copy_attributes: Annotated[bool, typer.Option(
        '--copy',
        '-c',
        help='Copy attributes from the input vector to the output vector.'
    )] = False
)-> None:
    """
    Sample raster values at point locations
    and save the results to a new vector file
    with CRS matching the input raster.
    """

    raster = read_geotiff(Path(input_raster))
    vector = read_shapefile(Path(input_vector))

    if not all(vector.geometry.type == 'Point'):
        raise ValueError('Input vector data must contain point geometry')

    sampled_table = point_sampling(
        raster=raster,
        vector=vector,
        include_attributes=copy_attributes
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