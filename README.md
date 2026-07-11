# Shen

Shen is a contract-oriented, DataFrame-first pricing library built on Polars and Pandera.

## Architecture

The public domain vocabulary is normalized around atomic instruments, tickets, ticket components, market data, and values:

- `shen.domain.contracts.instruments` contains atomic Pandera `DataFrameModel` terms keyed by `instrument_id`.
- `shen.domain.contracts.tickets` contains `Ticket`, `TicketComponent`, `Position`, and `validate_ticket_book`.
- `shen.domain.contracts.market` contains canonical market data frames. Instruments own requested dates; market frames expose market coordinates.
- `shen.pricing` contains market fetch declarations, pricers, and the lazy pricing engine.
- `shen.portfolio` derives component, ticket, position, MTM, PnL, and basis frames.
- `shen.risk` contains only implemented risk measures such as curve DV01.

Structured tickets are represented relationally: one ticket row and one or more component rows. Pricing dispatch is by atomic instrument `product_type`; signed `TicketComponent.quantity` is the only holding direction.

## Validation

`ShenFrame.validate` is the public boundary path. It applies declaration-ordered `@derived` transformations, materializes once, validates with Pandera, and returns a `LazyFrame`.

`ShenFrame.resolve` is the trusted internal path. It stays lazy and applies schema-level validation only; value-level checks are not fully executed until a public validation boundary.
