import argparse
import csv
import itertools
import operator
import os
from argparse import Namespace
from calendar import monthrange, isleap
from dataclasses import dataclass
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple

from gql import Client, gql
from gql.transport.aiohttp import AIOHTTPTransport
from graphql import DocumentNode
from pydantic import BaseModel, Field
from pydantic.alias_generators import to_camel

URL = "https://api.hark.eco"


class CamelCaseObject(BaseModel):
    class Config:
        alias_generator = to_camel
        populate_by_name = True


@dataclass
class PriceSpecification(CamelCaseObject):
    price: float
    price_currency: str
    vat: float
    price_excluding_vat: float


@dataclass
class EnergyRangeInWh(CamelCaseObject):
    range_from: int = Field(alias="from")
    range_to: int = Field(alias="to")


@dataclass
class CapacityStep(CamelCaseObject):
    range_in_wh: EnergyRangeInWh
    capacity_step_price: PriceSpecification


@dataclass
class MonthlyPeaksAtDifferentDays(CamelCaseObject):
    peaks_per_month: int
    capacity_steps: List[CapacityStep]
    price_above_capacity_steps: PriceSpecification


@dataclass
class CurrentRangePrice(CamelCaseObject):
    from_current: int
    to_current: int
    yearly_price: PriceSpecification


@dataclass
class PhaseVoltagePrices(CamelCaseObject):
    current_range_prices: List[CurrentRangePrice]
    yearly_price_above_current_ranges: PriceSpecification


@dataclass
class PhasePrices(CamelCaseObject):
    voltage230: PhaseVoltagePrices
    voltage400: PhaseVoltagePrices


@dataclass
class FuseAndVoltageSize(CamelCaseObject):
    one_phase_prices: PhasePrices
    three_phase_prices: PhasePrices


@dataclass
class DeliveryChargeEnergyTaxes(CamelCaseObject):
    electrical_power_tax: PriceSpecification
    energy_fund: PriceSpecification


@dataclass
class PricesAndTaxes(CamelCaseObject):
    price_including_energy_taxes: PriceSpecification
    price_excluding_energy_taxes: PriceSpecification
    taxes: DeliveryChargeEnergyTaxes


@dataclass
class ConstantPrice(CamelCaseObject):
    prices_and_taxes: PricesAndTaxes
    # rush_pricing: RushPricing


@dataclass
class HourRange(CamelCaseObject):
    hours_from: int = Field(alias="from")
    hours_until: int = Field(alias="until")


@dataclass
class PricesAndTaxesForPartOfDay(CamelCaseObject):
    hours: HourRange
    pricesAndTaxes: PricesAndTaxes


@dataclass
class DayAndNightPrices(CamelCaseObject):
    day: PricesAndTaxesForPartOfDay
    night: PricesAndTaxesForPartOfDay


@dataclass
class DifferentPricesDayAndNight(CamelCaseObject):
    prices: DayAndNightPrices
    weekend_prices_follow_night_prices: bool
    holiday_prices_follow_night_prices: bool


@dataclass
class MonthRange(CamelCaseObject):
    from_month: int = Field(alias="from")
    until_month: int = Field(alias="until")


@dataclass
class PricesForPartOfYearWithoutDayAndNight(CamelCaseObject):
    months: MonthRange
    pricesAndTaxes: PricesAndTaxes


@dataclass
class SeasonalPricesWithoutDayAndNight(CamelCaseObject):
    summer: PricesForPartOfYearWithoutDayAndNight
    winter: PricesForPartOfYearWithoutDayAndNight


@dataclass
class DifferentPricesSeasonal(CamelCaseObject):
    seasons: SeasonalPricesWithoutDayAndNight


@dataclass
class SelectableEnergyModel(CamelCaseObject):
    constant_price: Optional[ConstantPrice] = None
    different_prices_day_and_night: Optional[DifferentPricesDayAndNight] = None
    different_prices_seasonal: Optional[DifferentPricesSeasonal] = None


@dataclass
class CurrentRangeWithSelectable(CamelCaseObject):
    from_current: int
    to_current: int
    selectable_energy_model: SelectableEnergyModel


@dataclass
class SelectablePhaseVoltagePrices(CamelCaseObject):
    current_ranges_with_selectable: List[CurrentRangeWithSelectable]


@dataclass
class SelectablePhasePrices(CamelCaseObject):
    voltage230: SelectablePhaseVoltagePrices
    voltage400: SelectablePhaseVoltagePrices


@dataclass
class FuseBasedSelectable(CamelCaseObject):
    one_phase_prices: SelectablePhasePrices
    three_phase_prices: SelectablePhasePrices


@dataclass
class HourValue:
    time: datetime
    value_in_wh: int


@dataclass
class CapacityModelDetails:
    capacity_cost: float
    capacity_step: Optional[CapacityStep] = None
    tops: Optional[List[HourValue]] = None
    top_average: Optional[float] = None


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


def capacity_model_details_from_fuse_and_voltage_size_model(
    capacity_model: FuseAndVoltageSize, args: Namespace
) -> Optional[CapacityModelDetails]:
    model = phase_voltage_prices_from_model_and_extra_args(capacity_model, args)

    yearly_price = None
    for current_step in model.current_range_prices:
        if (
            args.current > current_step.from_current
            and args.current <= current_step.to_current
        ):
            yearly_price = current_step.yearly_price
    if yearly_price:
        days_in_month = find_days_in_month(args.date)
        days_in_year = find_days_in_year(args.date)
        capacity_cost = yearly_price.price * days_in_month / days_in_year
        return CapacityModelDetails(capacity_cost)
    else:
        print("Could not find current range")
        exit(1)


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


def energy_model_details_from_constant_price_model(
    hour_values: List[HourValue], energy_model: ConstantPrice
) -> EnergyModelDetails:
    hour_values_sum = sum([hv.value_in_wh for hv in hour_values])
    energy_cost_alone: float = (
        hour_values_sum
        * energy_model.prices_and_taxes.price_excluding_energy_taxes.price
        / 1000
    )
    power_tax = (
        hour_values_sum * energy_model.prices_and_taxes.taxes.electrical_power_tax.price
    ) / 1000
    energy_fund = (
        hour_values_sum * energy_model.prices_and_taxes.taxes.energy_fund.price
    ) / 1000
    return EnergyModelDetails(
        energy_cost=energy_cost_alone + power_tax + energy_fund,
        energy_cost_alone=energy_cost_alone,
        power_tax=power_tax,
        energy_fund=energy_fund,
    )


def energy_model_details_from_different_prices_seasonal_model(
    hour_values_sum: int,
    energy_model: DifferentPricesSeasonal,
    args: Namespace,
) -> EnergyModelDetails:
    month = args.date.month
    summer_start = energy_model.seasons.summer.months.from_month
    winter_start = energy_model.seasons.winter.months.from_month

    if month in range(summer_start, winter_start):
        prices_and_taxes = energy_model.seasons.summer.pricesAndTaxes
    else:
        prices_and_taxes = energy_model.seasons.winter.pricesAndTaxes

    energy_cost_alone: float = (
        hour_values_sum * prices_and_taxes.price_excluding_energy_taxes.price / 1000
    )
    power_tax = (
        hour_values_sum * prices_and_taxes.taxes.electrical_power_tax.price
    ) / 1000
    energy_fund = (hour_values_sum * prices_and_taxes.taxes.energy_fund.price) / 1000
    return EnergyModelDetails(
        energy_cost=energy_cost_alone + power_tax + energy_fund,
        energy_cost_alone=energy_cost_alone,
        power_tax=power_tax,
        energy_fund=energy_fund,
    )


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


def energy_model_details_from_fuse_based_selectable(
    hour_values: List[HourValue], energy_model: FuseBasedSelectable, args: Namespace
) -> Optional[EnergyModelDetails]:
    hour_values_sum = sum([hv.value_in_wh for hv in hour_values])
    model = selectable_phase_voltage_prices_from_model_and_extra_args(
        energy_model, args
    )

    for current_step in model.current_ranges_with_selectable:
        if (
            args.current > current_step.from_current
            and args.current <= current_step.to_current
        ):
            selectable_energy_model = current_step.selectable_energy_model
            selectable_dict = dict(selectable_energy_model)
            filtered_models = [m for m in selectable_dict.values() if m]
            if len(filtered_models) == 1:
                model = filtered_models[0]
            else:
                if args.selected_model is None:
                    print(
                        "The model is selectable, we need 'selected_model' parameter set"
                    )
                    exit(1)
                elif args.selected_model == "constant_price":
                    energy_model_constant_price: ConstantPrice = selectable_dict[
                        args.selected_model
                    ]
                    return energy_model_details_from_constant_price_model(
                        hour_values, energy_model_constant_price
                    )
                elif args.selected_model == "different_prices_seasonal":
                    energy_model_different_prices_seasonal: DifferentPricesSeasonal = (
                        selectable_dict[args.selected_model]
                    )
                    return energy_model_details_from_different_prices_seasonal_model(
                        hour_values_sum, energy_model_different_prices_seasonal, args
                    )
                else:
                    print("Selected model not found")
                    exit(1)


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


# Helper to dig into deeper structure based on extra cli arguments
def phase_voltage_prices_from_model_and_extra_args(
    energy_model: FuseAndVoltageSize, args: Namespace
) -> PhaseVoltagePrices:
    if args.phases == 3:
        model = energy_model.three_phase_prices
    else:
        model = energy_model.one_phase_prices
    if args.voltage == 400:
        model = model.voltage400
    else:
        model = model.voltage230
    if args.current is None:
        print("Missing parameter current")
        exit(1)
    return model


# Helper to dig into deeper structure based on extra cli arguments
def selectable_phase_voltage_prices_from_model_and_extra_args(
    energy_model: FuseBasedSelectable, args: Namespace
) -> SelectablePhaseVoltagePrices:
    if args.phases == 3:
        prices = energy_model.three_phase_prices
    else:
        prices = energy_model.one_phase_prices
    if args.voltage == 400:
        prices = prices.voltage400
    else:
        prices = prices.voltage230
    if args.current is None:
        print("Missing parameter current")
        exit(1)
    return prices


class HarkApi:
    def __init__(self, api_key: str):
        transport = AIOHTTPTransport(url=URL, headers={"X-API-KEY": api_key})

        self.client = Client(transport=transport, fetch_schema_from_transport=True)

    def execute(self, query: DocumentNode, variables: Dict[str, Any]) -> Dict[str, Any]:
        return self.client.execute(query, variable_values=variables)


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


def read_graphql_query_from_file(path: str) -> DocumentNode:
    with open(path, newline="") as file:
        return gql(file.read())


def group_by_day(hour_values: List[HourValue]) -> Dict[str, List[HourValue]]:
    hour_values_by_day = dict()
    grouped = itertools.groupby(hour_values, lambda hv: hv.time.date())
    for _date, values in grouped:
        hour_values_by_day[_date] = list(values)

    return hour_values_by_day


def find_top_hour_per_day(
    hour_values_by_day: Dict[str, List[HourValue]]
) -> List[HourValue]:
    return [
        sorted(hour_values, key=operator.attrgetter("value_in_wh"), reverse=True)[0]
        for hour_values in hour_values_by_day.values()
    ]


def find_highest_day_tops(
    top_hour_per_day: List[HourValue], number_of_top_hours: int
) -> List[HourValue]:
    return sorted(
        top_hour_per_day, key=operator.attrgetter("value_in_wh"), reverse=True
    )[:number_of_top_hours]


def find_average_of_tops(tops: List[HourValue]) -> float:
    if len(tops) < 1:
        print("No tops to average")
        exit(1)

    return sum([hv.value_in_wh for hv in tops]) / len(tops)


def split_day_and_night(
    hour_values: List[HourValue], day_start, night_start, weekend_as_night
) -> Tuple[List[int], List[int]]:
    dayprice_amounts: List[int] = []
    night_price_amounts: List[int] = []

    for hv in hour_values:
        if weekend_as_night and hv.time.isoweekday() in [6, 7]:
            night_price_amounts.append(hv.value_in_wh)
        elif hv.time.hour not in range(day_start, night_start):
            night_price_amounts.append(hv.value_in_wh)
        else:
            dayprice_amounts.append(hv.value_in_wh)

    return (dayprice_amounts, night_price_amounts)


def find_days_in_month(_date: date):
    return monthrange(_date.year, _date.month)[1]


def find_days_in_year(_date: date):
    return 365 + isleap(_date.year)


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

    print(capacity_model_details_from_capacity_model(hour_values, dso_model, args))
    print(energy_model_details_from_energy_model(hour_values, dso_model, args))


if __name__ == "__main__":
    main()
