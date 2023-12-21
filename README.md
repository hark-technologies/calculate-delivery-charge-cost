# Use delivery charge API and Elhub data to compute grid cost invoice data

This is an example of how we can use the Hark API to fetch DSO models with pricing info, and apply this to consumption data downloaded from Elhub.

To set up the Python project, we start by creating a virtual environment and install necessary packages.
We need a few external packages to use Graphql and convert returned data to Python types.

```sh
python3 -m venv .venv
source .venv/activate # or activate.fish or similar
pip install "gql[all]"
pip install graphql
pip install pydantic

touch src/main.py
```

Then we start with something minimal in `src/main.py`, and add more code to the file as we explain what it contains:

```python
import os


URL = "https://api.hark.eco"


def main():
    api_key = os.environ.get("API_KEY")
    if api_key is None or api_key == "":
        print("API_KEY not set")
        exit(1)
    args = parse_arguments()

    hark = HarkApi(api_key)
    dso_query = read_graphql_query_from_file("src/dso_model_query.graphql")
    dso_model = hark.execute(dso_query, dict(id=args.id, date=args.date.isoformat()))[
        "deliveryChargeModel"
    ]

    hour_values = read_hour_values_from_csv(args.path)

    capacity_model_details = capacity_model_details_from_capacity_model(hour_values, dso_model, args)
    energy_model_details = energy_model_details_from_energy_model(hour_values, dso_model, args)
    print(capacity_model_details)
    print(energy_model_details)


if __name__ == "__main__":
    main()
```

To use the API, you need an `API_KEY`. We read this from the environment so that you can fetch it from a password store or similar.

Then we parse arguments and set up a Graphql client. The Graphql query is huge, and we read it from a file before we use it to query the API. We use the `deliveryChargeModel` from the result.

Elhub data is read from a CSV file, and we are then using these data combined with the DSO models and command line arguments to compute and present representations of what the customer paid in Nettleie.

## Parsing arguments

Add the next few lines to the top, and `parse_arguments` function above the `main` function

```python
import argparse
```

```python
def parse_arguments() -> Namespace:
    parser = argparse.ArgumentParser("Beregn nettleie")
    parser.add_argument(
        "--meteringvalues", help="Path to Elhub data csv", dest="path", required=True
    )
    parser.add_argument(
        "--id",
        help="Hark nettleiemodell id, (uppercase string)",
        required=True,
    )
    parser.add_argument(
        "--date",
        help="ISO string",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
        required=True,
    )
    parser.add_argument("--voltage", type=int)
    parser.add_argument("--phases", type=int)
    parser.add_argument("--current", type=int)
    parser.add_argument(
        "--selected-model", choices=["constant_price", "different_prices_seasonal"]
    )
    return parser.parse_args()
```

`datetime` is needed, so add this import (`date` is needed later)

```python
from datetime import datetime, date
```

Here, the `date`, `id` and `meteringvalues` are mandatory and needed for all model types.

- `date` is used for invoice time span and to query the correct version of the DSO model.
- `id` is our internal ID, an uppercase value that represents the provider (`ELVIA`, `LEDE`, `TENSIO_TS` are some examples)
- `meteringvalues` is the path to the Elhub CSV file (that has a name starting with "meteringvalues")

The other, optional values will be described when we get to one of the more complicated models.

## Using the Hark API

Repeating the next section from `main`:

```python
    hark = HarkApi(api_key)
    dso_query = read_graphql_query_from_file("src/dso_model_query.graphql")
    dso_model = hark.execute(dso_query, dict(id=args.id, date=args.date.isoformat()))[
        "deliveryChargeModel"
    ]
```

The `HarkApi` is defined as:

```python
class HarkApi:
    def __init__(self, api_key: str):
        transport = AIOHTTPTransport(url=URL, headers={"X-API-KEY": api_key})

        self.client = Client(transport=transport, fetch_schema_from_transport=True)

    def execute(self, query: DocumentNode, variables: Dict[str, Any]) -> Dict[str, Any]:
        return self.client.execute(query, variable_values=variables)
```

You also need these imports:

```python
from gql import Client, gql
from gql.transport.aiohttp import AIOHTTPTransport
from graphql import DocumentNode
```

We also use typing, and we can import this now as well.

```python
from typing import Any, Dict, List, Optional, Tuple
```

Reading the Graphql query is pretty straight forward, but the content is parsed as a Graphql `DocumentNode` by the imported `gql` function.

```python
def read_graphql_query_from_file(path: str) -> DocumentNode:
    with open(path, newline="") as file:
        return gql(file.read())
```

## Reading Elhub data

Reading the CSV data from Elhub is also pretty normal, except that we need to remember to skip the first line, containing headers.

```python
def read_hour_values_from_csv(path: str) -> List[HourValue]:
    hour_values = []
    with open(path, newline="") as csvfile:
        hour_value_reader = csv.reader(csvfile)
        # Skip header line
        next(hour_value_reader)

        for row in hour_value_reader:
            hour_value = HourValue(
                datetime.strptime(row[0], "%d.%m.%Y %H:%M"),
                int(float(row[2].replace(",", ".")) * 1000),
            )
            hour_values.append(hour_value)
    return hour_values
```

Of course this is using the `csv` module, so you need to add another import:
```python
import csv
```

When parsing the data into our own `HourValue` class, we parse the date with the format in the CSV file. The consumption values in the file are stored as kWh with three decimals, and we convert it to Wh, as it is easier to work with integers.

Our `HourValue` class is defined as:

```python
@dataclass
class HourValue:
    time: datetime
    value_in_wh: int
```

The `@dataclass` decorator is a handy way of getting constructors and other helper functions automatically.


## Capacity models

`capacity_model_details_from_capacity_model` is a function that does computations on the capacity parts.
It will in turn run different function based on the different capacity models.

```python
def capacity_model_details_from_capacity_model(
    hour_values: List[HourValue], dso_model: Any, args: Namespace
) -> Optional[CapacityModelDetails]:
    if dso_model["capacityModel"]["__typename"] == "MonthlyPeaksAtDifferentDays":
        capacity_model = MonthlyPeaksAtDifferentDays.model_validate(
            dso_model["capacityModel"]
        )
        return capacity_model_details_from_monthly_peaks_at_different_days_model(
            hour_values, capacity_model
        )
    elif dso_model["capacityModel"]["__typename"] == "FuseAndVoltageSize":
        capacity_model = FuseAndVoltageSize.model_validate(dso_model["capacityModel"])
        return capacity_model_details_from_fuse_and_voltage_size_model(
            capacity_model, args
        )
```

All models are converted with the `model_validate` function from `pydantic`.
This is used to get Python types from the returned Graphql JSON.

Let's just have a quick look at how this works before looking at the capacity models in detail.

```python
@dataclass
class MonthlyPeaksAtDifferentDays(CamelCaseObject):
    peaks_per_month: int
    capacity_steps: List[CapacityStep]
    price_above_capacity_steps: PriceSpecification
```

The `int` and `List` are Python types, while `PriceSpecification` is a Hark type, defined in the same way as we see here.

We have created a common superclass named `CamelCaseObject` to use instead of the plain `BaseModel` from `pydantic` to avoid repeating the `Config` class inside all of our types.

```python
class CamelCaseObject(BaseModel):
    class Config:
        alias_generator = to_camel
        populate_by_name = True
```

### Capacity model: MonthlyPeaksAtDifferentDays

This model has a set of capacity steps, and it is usually the average of the three hours of highest consumption, measured at different days that decides the steps.

```python
def capacity_model_details_from_monthly_peaks_at_different_days_model(
    hour_values: List[HourValue],
    capacity_model: MonthlyPeaksAtDifferentDays,
) -> CapacityModelDetails:
    number_of_tops = capacity_model.peaks_per_month
    hour_values_by_day = group_by_day(hour_values)
    top_hour_per_day = find_top_hour_per_day(hour_values_by_day)
    top_hours_for_a_month = find_highest_day_tops(top_hour_per_day, number_of_tops)
    average_of_tops = find_average_of_tops(top_hours_for_a_month)

    capacity_step: Optional[CapacityStep] = None
    for step in capacity_model.capacity_steps:
        range_from = step.range_in_wh.range_from
        range_to = step.range_in_wh.range_to

        if average_of_tops >= range_from and average_of_tops < range_to:
            capacity_step = step
            break

    price_specification = (
        capacity_model.price_above_capacity_steps
        if capacity_step is None
        else capacity_step.capacity_step_price
    )

    capacity_cost = price_specification.price
    return CapacityModelDetails(
        capacity_cost,
        capacity_step,
        tops=top_hours_for_a_month,
        top_average=average_of_tops,
    )
```

We first get the number of tops from the capacity model.
Then we group the consumption data by day and find the top hour per day, before ordering these by the highest values and only returning the number specified in the model.
We then take the average of these and use this value when looping through the steps to find the matching capacity step.

If the average don't match any steps, the consumer has exceeded the steps and we have to use the `price_above_capacity_steps` when determining the price.

All prices are returned as a `PriceSpecification`, to make it easy to get the prices including and excluding VAT.

```python
@dataclass
class PriceSpecification(CamelCaseObject):
    price: float
    price_currency: str
    vat: float
    price_excluding_vat: float
```

We need to get into the implementation details for this model.

To `group_by_day` we use the built-in `itertools` function `group_by`:

```python
def group_by_day(hour_values: List[HourValue]) -> Dict[str, List[HourValue]]:
    hour_values_by_day = dict()
    grouped = itertools.groupby(hour_values, lambda hv: hv.time.date())
    for _date, values in grouped:
        hour_values_by_day[_date] = list(values)

    return hour_values_by_day
```

And to find the top hour per day, we sort by the values (using `sorted` and `operator.attrgetter`) for each day and pick the highest.

```python
def find_top_hour_per_day(
    hour_values_by_day: Dict[str, List[HourValue]]
) -> List[HourValue]:
    return [
        sorted(hour_values, key=operator.attrgetter("value_in_wh"), reverse=True)[0]
        for hour_values in hour_values_by_day.values()
    ]
```

To find the highest day tops for the month, we sort these values by value, and take the (normally 3) top hours.

```python
def find_highest_day_tops(
    top_hour_per_day: List[HourValue], number_of_top_hours: int
) -> List[HourValue]:
    return sorted(
        top_hour_per_day, key=operator.attrgetter("value_in_wh"), reverse=True
    )[:number_of_top_hours]
```

Then we compute the average the values.

```python
def find_average_of_tops(tops: List[HourValue]) -> float:
    if len(tops) < 1:
        print("No tops to average")
        exit(1)

    return sum([hv.value_in_wh for hv in tops]) / len(tops)
```

For this we need to import a few more modules:

```python
import itertools
import operator
```

All the capacity model functions return an instance of `CapacityModelDetails`, but some of the fields are optional.

```python
@dataclass
class CapacityModelDetails:
    capacity_cost: float
    capacity_step: Optional[CapacityStep] = None
    tops: Optional[List[HourValue]] = None
    top_average: Optional[float] = None
```

### Other capacity models

The final script also contains handling of the `FuseAndVoltageBased` capacity model.
This and the ones remaining are used by one or two providers each.

## Energy models

Energy models are more complicated, and this is also true for the data they return.
`EnergyModelDetails` returned by `energy_model_details_from_energy_model` and the individual model parsers reflect that.

```python
@dataclass
class EnergyModelDetails:
    energy_cost: float
    power_tax: float
    energy_fund: float
    energy_cost_alone: Optional[float] = None
    day_cost: Optional[float] = None
    night_cost: Optional[float] = None
    day_power_tax: Optional[float] = None
    night_power_tax: Optional[float] = None
    day_energy_fund: Optional[float] = None
    night_energy_fund: Optional[float] = None
```

And the top function for energy model handling:

```python
def energy_model_details_from_energy_model(
    hour_values: List[HourValue],
    dso_model: Any,
    args: Namespace,
) -> Optional[EnergyModelDetails]:
    if dso_model["energyModel"]["__typename"] == "ConstantPrice":
        energy_model = ConstantPrice.model_validate(dso_model["energyModel"])
        return energy_model_details_from_constant_price_model(hour_values, energy_model)
    elif dso_model["energyModel"]["__typename"] == "DifferentPricesDayAndNight":
        energy_model = DifferentPricesDayAndNight.model_validate(
            dso_model["energyModel"]
        )
        return energy_model_details_from_different_prices_day_and_night_model(
            hour_values, energy_model
        )
    elif dso_model["energyModel"]["__typename"] == "FuseBasedSelectable":
        energy_model = FuseBasedSelectable.model_validate(dso_model["energyModel"])
        return energy_model_details_from_fuse_based_selectable(
            hour_values, energy_model, args
        )
```
 

The simplest energy model is the one we call `ConstantPrice`, but for now we take one that is more used and a bit more difficult to implement, the one we call `DifferentPricesDayAndNight`.

### Energy model: DifferentPricesDayAndNight

This model is used by both `ELVIA` and `TENSIO_TS`, so it was easy for us to find data to test it.
It is a bit long, but the length is mainly because the different providers use different fields in their invoices.

```python
def energy_model_details_from_different_prices_day_and_night_model(
    hour_values: List[HourValue],
    energy_model: DifferentPricesDayAndNight,
) -> EnergyModelDetails:
    day_start = energy_model.prices.day.hours.hours_from
    night_start = energy_model.prices.night.hours.hours_from
    weekend_as_night = energy_model.weekend_prices_follow_night_prices
    (daytime_amounts, nighttime_amounts) = split_day_and_night(
        hour_values, day_start, night_start, weekend_as_night
    )
    energy_cost_day = (
        sum(daytime_amounts)
        * energy_model.prices.day.pricesAndTaxes.price_excluding_energy_taxes.price
        / 1000
    )
    energy_cost_night = (
        sum(nighttime_amounts)
        * energy_model.prices.night.pricesAndTaxes.price_excluding_energy_taxes.price
        / 1000
    )
    power_tax_day = (
        sum(daytime_amounts)
        * energy_model.prices.day.pricesAndTaxes.taxes.electrical_power_tax.price
        / 1000
    )
    power_tax_night = (
        sum(nighttime_amounts)
        * energy_model.prices.night.pricesAndTaxes.taxes.electrical_power_tax.price
        / 1000
    )
    power_tax = power_tax_day + power_tax_night
    energy_fund_day = (
        sum(daytime_amounts)
        * energy_model.prices.day.pricesAndTaxes.taxes.energy_fund.price
        / 1000
    )
    energy_fund_night = (
        sum(nighttime_amounts)
        * energy_model.prices.night.pricesAndTaxes.taxes.energy_fund.price
        / 1000
    )
    energy_fund = energy_fund_day + energy_fund_night
    energy_cost_alone = energy_cost_day + energy_cost_night
    energy_cost = energy_cost_alone + power_tax + energy_fund

    return EnergyModelDetails(
        energy_cost=energy_cost,
        energy_cost_alone=energy_cost_alone,
        day_cost=energy_cost_day,
        night_cost=energy_cost_night,
        day_power_tax=power_tax_day,
        night_power_tax=power_tax_night,
        power_tax=power_tax,
        day_energy_fund=energy_fund_day,
        night_energy_fund=energy_fund_night,
        energy_fund=energy_fund,
    )
```

We start by finding the hours that is defined by the model as day and night.
We also check if all consumption during weekends should be billed as night. (This is different for Elvia and Tensio-Ts).
There is also a setting for if holidays should be billed as night, but we ignore it at this time (none of Elvia and Tensio-Ts do this).

We have a helper function that will sort the consumption data into a list for day and a list for night (and weekend), and we will show it below.

Computing the cost of the energy is done by taking the sum of the values, and multiplying by the cost per kWh excluding energy taxes, for days and nights.
The same is done for each of the two energy taxes, to get all data to compare to invoices.
We then sum the different night values and day values, and then the total.

## A note about VAT and rounding

We simplify a bit by using prices including VAT in this example. (Everywhere we use the field `price`, there is also a `priceWithoutVat`/`price_without_vat`.)
There may be some rounding errors as both we and the providers round some of the values.
(Prices are rounded to milli-øre from our API, as a best effort, since it varies if price data from the providers are given with VAT and other taxes - when testing a few values are one øre off). It also seems the providers are rounding differently, some using bankers rounding.
