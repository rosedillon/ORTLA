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
    reversal = [{"tiploc_code": "HMPDNPK", "crs_code": "HMD", "arr": "1000"},
                {"tiploc_code": "EBOURNE", "crs_code": "EBN", "arr": "1005"},
                {"tiploc_code": "HMPDNPK", "crs_code": "HMD", "arr": "1015"}]
    assert len(_merge_calls(reversal)) == 3



def test_header_ends():
    from ortla.wtt_workbook import _header_ends
    names = {"ASHINGTON": ("ASHGTN", "ASL"), "NEWCASTLE": ("NWCSTLE", "NCL"), "HEATON SOUTH JN": ("HTONSJN", None)}
    calls = [{"tiploc_code": "HTONSJN", "crs_code": None, "pass": "1436"},
             {"tiploc_code": "NWCSTLE", "crs_code": "NCL", "arr": "1442"}]
    header = {"origin": "ASHINGTON\n14:07", "destination": "NEWCASTLE\n14:42"}
    assert _header_ends(header, calls, names) == [
        {"tiploc_code": "ASHGTN", "crs_code": "ASL", "row": -1, "dep": "1407", "stops": True}]


def test_merge_across_midnight_and_junctions():
    from ortla.wtt_workbook import _merge_calls
    midnight = [{"tiploc_code": "BHAMNWS", "crs_code": "BHM", "arr": "2359"},
                {"tiploc_code": "BHAMNWS", "crs_code": "BHM", "dep": "0002"},
                {"tiploc_code": "SMETHWK", "crs_code": "SGB", "dep": "0008"}]
    merged = _merge_calls(midnight)
    assert len(merged) == 2
    assert [c for c in merged if c["tiploc_code"] == "BHAMNWS"][0] | {} == {
        "tiploc_code": "BHAMNWS", "crs_code": "BHM", "arr": "2359", "dep": "0002", "pass": None, "stops": False}
    junction = [{"tiploc_code": "CMTHN", "crs_code": "CMN", "arr": "0012"},
                {"tiploc_code": "CMTHNJ", "crs_code": None, "pass": "0014"},
                {"tiploc_code": "CMTHN", "crs_code": "CMN", "arr": "0016"}]
    assert len(_merge_calls(junction)) == 2


def test_order_loop_back_to_origin():
    calls = [{"tiploc_code": "WATRLOO", "dep": "2257"}, {"tiploc_code": "CLPHMJN", "dep": "2305"},
             {"tiploc_code": "KINGSTN", "dep": "2330"}, {"tiploc_code": "WATRLOO", "arr": "0016"}]
    run = ["2257", "2305", "2330", "0016"]
    assert [c.get("dep") or c.get("arr") for c in _order_calls(calls, "WATRLOO", "2257")] == run


def test_date_groups():
    from ortla.wtt_workbook import _date_groups
    part = lambda dates: {"header": {"dates": dates}, "calls": []}
    a, b, c = part("18/05/2026 to 18/09/2026"), part("21/09/2026 to 11/12/2026"), part("18/05/2026 to 11/12/2026")
    assert [len(g) for g in _date_groups([a, b])] == [1, 1]   # two versions
    assert [len(g) for g in _date_groups([a, b, c])] == [3]   # one book spans both: one schedule
