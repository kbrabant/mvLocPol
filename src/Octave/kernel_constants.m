% Calculate the constant to make the kernel a density
function const = kernel_constants(kernel, d, detH)
    switch kernel
        case 'Epa'
            cd = ((d + 2) * gamma(d/2 + 1)) / (2 * pi^(d/2));
        case 'gauss'
            cd = (2 * pi)^(-d / 2);
        case 'bimodgauss'
            cd = 2 / (pi^(d/2) * d);
        case 'triangle'
            cd = d * (d + 1) * gamma(d / 2) / (2 * pi^(d/2));
        otherwise
            error('Unsupported kernel.');
    end
    const = cd / detH;
end


