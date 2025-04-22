from radarx.fundamentals import timing


def test_pulse_repetition_interval():
    prf = 1000  # Hz
    pri = timing.compute_pulse_repetition_interval(prf)
    assert pri == 1 / prf


def test_duty_cycle():
    pw = 1e-6  # 1 microsecond
    prf = 1000  # Hz
    dc = timing.compute_duty_cycle(pw, prf)
    assert 0 < dc < 1


def test_max_unambiguous_range():
    prf = 1000  # Hz
    rmax = timing.compute_max_unambiguous_range(prf)
    assert rmax > 0


def test_max_unambiguous_velocity():
    prf = 1000  # Hz
    wl = 0.03  # 3 cm
    vmax = timing.compute_max_unambiguous_velocity(prf, wl)
    assert vmax > 0


def test_blind_range():
    pulse_width = 1e-6  # 1 microsecond
    blind = timing.compute_blind_range(pulse_width)
    assert blind > 0


def test_compute_prf():
    pulse_width = 1e-6  # 1 microsecond
    duty_cycle = 0.1
    prf = timing.compute_prf(pulse_width, duty_cycle)
    expected_prf = duty_cycle / pulse_width
    assert prf == expected_prf
