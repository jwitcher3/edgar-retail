COMPANY_TICKERS_JSON = "https://www.sec.gov/files/company_tickers.json"
COMPANY_TICKERS_EXCHANGE_JSON = "https://www.sec.gov/files/company_tickers_exchange.json"
SUBMISSIONS_JSON = "https://data.sec.gov/submissions/CIK{cik10}.json"
COMPANY_FACTS_JSON = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json"


def accession_no_dashes(accession: str) -> str:
    return accession.replace("-", "")


def filing_primary_doc_url(cik: int, accession: str, primary_doc: str) -> str:
    return (
        f"https://www.sec.gov/Archives/edgar/data/{cik}/"
        f"{accession_no_dashes(accession)}/{primary_doc}"
    )
