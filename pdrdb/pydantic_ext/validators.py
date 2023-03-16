import phonenumbers
from phonenumbers import COUNTRY_CODE_TO_REGION_CODE

MAX_COUNTRY_CODE_LEN = 3


def parse_phone_number(p, region='US'):
    try:
        if p[:1] != '+':
            p = f'+{p}'
        p_obj = phonenumbers.parse(p, region)
        if phonenumbers.is_valid_number(p_obj):
            return p_obj
        raise ValueError('Invalid phone number')
    except phonenumbers.NumberParseException as e:
        raise ValueError(str(e))


def validate_phone_number(value, strict_e164=None):
    if strict_e164:
        return get_phone_number_using_country_code(value)
    return parse_phone_number(value)


def get_phone_number_using_country_code(number):
    if number[0] == '+':
        number = number[1:]
    country_code, national_number = get_country_code_and_rest_of_number(number)
    try:
        p_obj = phonenumbers.PhoneNumber(
            country_code=country_code,
            national_number=national_number,
        )
        if phonenumbers.is_valid_number(p_obj):
            return p_obj
    except Exception:
        pass
    raise ValueError('Invalid phone number')


def get_country_code_and_rest_of_number(number: str):
    """
    :param number:  Number without plus sign
    :return: (country_code, rest_of_number)
    """
    if len(number) == 0 or number[0] == '0':
        # Country codes do not begin with a '0'.
        return (0, number)
    for ii in range(1, min(len(number), MAX_COUNTRY_CODE_LEN) + 1):
        try:
            country_code = int(number[:ii])
            if country_code in COUNTRY_CODE_TO_REGION_CODE:
                return (country_code, number[ii:])
        except Exception:
            pass
    return (0, number)
