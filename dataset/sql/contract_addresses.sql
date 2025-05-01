SELECT
  receipt_contract_address,
  TIMESTAMP_TRUNC (block_timestamp, DAY) AS day
FROM
  `bigquery-public-data.crypto_ethereum.transactions`
WHERE
  (
    TIMESTAMP_TRUNC (block_timestamp, DAY) BETWEEN TIMESTAMP ("2022-06-1") AND TIMESTAMP  ("2024-06-30")
  )
  AND to_address IS NULL
  AND receipt_status != 0;