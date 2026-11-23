from ortla.wtt_workbook import _days_mask, _order_calls


def test_running_day_codes():
    assert _days_mask("SX") == "1111100"
    assert _days_mask("SUO") == "0000001"
    assert _days_mask("TWThO") == "0111000"
    assert _days_mask("TThO") == "0101000"
    assert _days_mask("WSX") == "1101100"
    assert _days_mask("ThFSO") == "0001110"
    assert _days_mask("MSO") == "1000010"
    assert _days_mask("EWD") == "1111110"
    assert _days_mask("") == "1111111"


def test_order_past_midnight():
    calls = [{"tiploc_code": t, "dep": d} for t, d in
             [("EALINGB", "0002"), ("PADTON", "0010"), ("RDNGSTN", "2317"), ("SLOUGH", "2339")]]
    run = ["RDNGSTN", "SLOUGH", "EALINGB", "PADTON"]
    assert [c["tiploc_code"] for c in _order_calls(calls, "RDNGSTN")] == run
    assert [c["tiploc_code"] for c in _order_calls(calls, None)] == run
    assert [c["tiploc_code"] for c in _order_calls(calls[2:], None)] == ["RDNGSTN", "SLOUGH"]


def test_activity_codes():
    from ortla.wtt_workbook import _parse_time
    assert _parse_time("15T45½") == ("1545", True)
    assert _parse_time("1557") == ("1557", False)
    assert _parse_time("15/48") == ("1548", False)
    assert _parse_time("06T RM24") == ("0624", True)
    assert _parse_time("23TF-U37") == ("2337", True)
    assert _parse_time("00R43") == ("0043", True)
    assert _parse_time("18RM34") == ("1834", False)
    assert _parse_time("05OPRM01") == ("0501", False)
    assert _parse_time("12A31") == ("1231", False)
    assert _parse_time("14RR04") == ("1404", False)
    assert _parse_time("06U49½") == ("0649", True)
    assert _parse_time("..") is None


def test_merge_keeps_reversal_calls():
    from ortla.wtt_workbook import _merge_calls
    # a station printed at the boundary of two books is one call
    boundary = [{"tiploc_code": "LEWES", "arr": "1000"}, {"tiploc_code": "LEWES", "dep": "1002"},
                {"tiploc_code": "FALMER", "arr": "1010"}]
    assert len(_merge_calls(boundary)) == 2
    # reversing at Eastbourne calls at Hampden Park twice
    reversal = [{"tiploc_code": "HMPDNPK", "arr": "1000"}, {"tiploc_code": "EBOURNE", "arr": "1005"},
                {"tiploc_code": "HMPDNPK", "arr": "1015"}]
    assert len(_merge_calls(reversal)) == 3

