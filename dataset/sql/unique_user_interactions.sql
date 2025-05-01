WITH
  # All transactions that involve a contract in the `contracts.all` table
  contract_transactions AS (
    SELECT
      input,
      to_address,
      from_address,
      receipt_status,
      block_timestamp,
      gas AS gas_used,
      nonce AS creator_nonce,
      receipt_contract_address,
      (LENGTH(input) / 2 - 1) AS input_data_length
    FROM
      `bigquery-public-data.crypto_ethereum.transactions`
    WHERE
      to_address IN (SELECT contract_address FROM `lookahead-zju.contracts.all`)
      OR receipt_contract_address IN (SELECT contract_address FROM `lookahead-zju.contracts.all`)
  ),

  # Of all the transactions involving a contract in the `contracts.all` table, find the ones that are contract deployment transactions
  contract_deployment_transactions AS (
    SELECT
      gas_used,
      creator_nonce,
      input_data_length,
      receipt_contract_address AS contract_address
    FROM
      contract_transactions
    WHERE
      to_address IS NULL
  )

# Of all the transactions involving a contract in the `contracts.all` table, find contract interaction transactions that are:
# - Not a contract deployment transaction
# - Between 2022-06-01 and 2024-08-01
# - Not a transfer transaction
# - Not a failed transaction
# Then count the number of unique users (from_addresses), and join with the `contract_deployment_transactions` table to get the creator_nonce, gas_used, and input_data_length
SELECT
  contract_txns.to_address AS contract_address,
  COUNT(DISTINCT contract_txns.from_address) AS unique_user_count,
  deployment_txns.gas_used AS gas_used,
  deployment_txns.creator_nonce AS creator_nonce,
  deployment_txns.input_data_length AS input_data_length
FROM
  contract_transactions contract_txns
JOIN
  contract_deployment_transactions deployment_txns
ON
  contract_txns.to_address = deployment_txns.contract_address
WHERE
  contract_txns.to_address IN (SELECT contract_address FROM `lookahead-zju.contracts.all`)
  AND TIMESTAMP_TRUNC(contract_txns.block_timestamp, DAY) BETWEEN TIMESTAMP("2022-06-01") AND TIMESTAMP("2024-08-01")
  AND contract_txns.input IS NOT NULL # Filter out transfer txs
  AND contract_txns.receipt_status = 1 # Filter out failed txns
GROUP BY
  contract_address, creator_nonce, gas_used, input_data_length
ORDER BY
  unique_user_count DESC;