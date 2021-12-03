class SelfSupervisedCompose:
    """
    Composes several transforms together.
    similar to torchvision.transforms.Compse.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample, target):
        for t in self.transforms:
            sample, target = t(sample, target)
        return (sample, target)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
