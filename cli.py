from pathlib import Path
from typing import Annotated

import typer

from sdb.io import read_geotiff, read_shapefile, write_geotiff, write_shapefile
from sdb.preprocessing import reproject_vector
from sdb.utils import median_filter, point_sampling

cli_app = typer.Typer(no_args_is_help=True)

@cli_app.command(
    name='median-filter',
    no_args_is_help=True,
)
def median_filter_cli(
    input_file: Annotated[str, typer.Argument(
        help="Path to the input raster file.",
    )],
    output_file: Annotated[str, typer.Argument(
        help="Path to the output raster file."
    )],
    filter_size: Annotated[int, typer.Option(
        "--filter-size",
        "-f",
        help="Size of the median filter window. "
        "Must be >= 3 and odd."
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
        help="Path to the input raster file."
    )],
    input_vector: Annotated[str, typer.Argument(
        help="Path to the input vector file containing point locations."
    )],
    output_vector: Annotated[str, typer.Argument(
        help="Path to the output vector file where sampled results will be saved."
    )]
)-> None:
    """
    Sample raster values at point locations
    and save the results to a new vector file.
    """

    raster = read_geotiff(Path(input_raster))
    vector = read_shapefile(Path(input_vector))

    # check if the vector data contains point geometry
    if not all(vector.geometry.type == 'Point'):
        raise ValueError('Input vector data must contain point geometry')

    new_vector = reproject_vector(
        raster=raster,
        vector=vector
    )

    sampled_table = point_sampling(
        raster=raster,
        x=new_vector.geometry.x,
        y=new_vector.geometry.y
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