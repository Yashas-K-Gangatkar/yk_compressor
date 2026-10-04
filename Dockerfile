FROM rust:1.97 as builder
WORKDIR /app
COPY . .
RUN cargo build --release --bin cloud_api

FROM debian:bookworm-slim
WORKDIR /app
COPY --from=builder /app/target/release/cloud_api .
EXPOSE 3000
CMD ["./cloud_api"]
