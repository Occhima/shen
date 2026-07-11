# Concepts

## Instrument

An instrument is the smallest independently priceable economic payoff in Shen. It owns atomic economic terms and dates required by its formula and produces a unit value.

## Ticket and component

A ticket is a booking event. `TicketComponent` connects a ticket to an atomic instrument and carries the signed quantity and quantity unit. A simple ticket has one component; a structured ticket has multiple components.

## Linear composition

If a ticket value can be reconstructed as `sum(component_quantity * component_unit_value)`, it is modeled as ticket components, not as a composite instrument or composite pricer.

## Market

A market is a validated collection of canonical market-data frames at a valuation date. Market frames expose available observation coordinates; instruments supply requested dates.
