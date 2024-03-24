import os


def cyr2latin(path, new_path):
    """
    Переименовывает файлы с изображениями ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif') из path
    (включая поддиректории), содержащие символы кириллицы и некоторое другие, в соответствии с legend dict object
    и копирует их в new_path. Структура поддиректорий не сохраняется. Если исходные файлы, находящиеся в разных
    поддиректориях имеют одинаковое наименование, то ко второму и последующим добавляется суффикс в формате "_hhhh",
    где 'hh' - случайное шестнадцатеричное число

    :param path: Путь, где лежат исходные файлы
    :param new_path: Путь, где будут лежать переименованные файлы
    """

    def convert(letter, dic):
        for i, j in dic.items():
            letter = letter.replace(i, j)
        return letter

    for root, d_names, f_names in os.walk(path):
        for f_name in f_names:
            if not f_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                print(f"{f_name} is skipped as '.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif' only are "
                      f"being precessed")
                continue
            f_name_new = convert(f_name, legend)
            file_path = os.path.join(root, f_name)
            file_new_path = os.path.join(new_path or path, f_name_new)
            if os.path.exists(file_new_path):
                f, ext = os.path.splitext(file_new_path)
                f = f + "_" + os.urandom(2).hex()
                file_new_path = f + ext

            os.rename(file_path, file_new_path)
            print(file_path, 'переименован в ', file_new_path)


legend = {
    '—': '-',
    ' ': '_',
    ',': '',
    'а': 'a',
    'б': 'b',
    'в': 'v',
    'г': 'g',
    'д': 'd',
    'е': 'e',
    'ё': 'yo',
    'ж': 'zh',
    'з': 'z',
    'и': 'i',
    'й': 'y',
    'к': 'k',
    'л': 'l',
    'м': 'm',
    'н': 'n',
    'о': 'o',
    'п': 'p',
    'р': 'r',
    'с': 's',
    'т': 't',
    'у': 'u',
    'ф': 'f',
    'х': 'h',
    'ц': 'c',
    'ч': 'ch',
    'ш': 'sh',
    'щ': 'such',
    'ъ': 'y',
    'ы': 'y',
    'ь': "'",
    'э': 'e',
    'ю': 'yu',
    'я': 'ya',

    'А': 'A',
    'Б': 'B',
    'В': 'V',
    'Г': 'G',
    'Д': 'D',
    'Е': 'E',
    'Ё': 'Yo',
    'Ж': 'Zh',
    'З': 'Z',
    'И': 'I',
    'Й': 'Y',
    'К': 'K',
    'Л': 'L',
    'М': 'M',
    'Н': 'N',
    'О': 'O',
    'П': 'P',
    'Р': 'R',
    'С': 'S',
    'Т': 'T',
    'У': 'U',
    'Ф': 'F',
    'Х': 'H',
    'Ц': 'Ts',
    'Ч': 'Ch',
    'Ш': 'Sh',
    'Щ': 'Such',
    'Ъ': 'Y',
    'Ы': 'Y',
    'Ь': "'",
    'Э': 'E',
    'Ю': 'Yu',
    'Я': 'Ya',
}
